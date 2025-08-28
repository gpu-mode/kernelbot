import asyncio
import contextlib
import datetime as dt
import logging
from dataclasses import dataclass

from libkernelbot.backend import KernelBackend
from libkernelbot.consts import SubmissionMode
from libkernelbot.report import MultiProgressReporter, RunProgressReporter, RunResultReport
from libkernelbot.submission import ProcessedSubmissionRequest
from libkernelbot.utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class JobItem:
    job_id: int
    sub_id: int
    req: ProcessedSubmissionRequest
    mode: SubmissionMode


# Periodicaly update the last heartbeat time for the submission job in submission_job_status table
HEARTBEAT_SEC = 15  # heartbeat interval 15s
# HARD_TIMEOUT_SEC [3hours]:if a submission is not completed within the hard timeout,
# it will be marked as failed in submission_job_status table
HARD_TIMEOUT_SEC = 60 * 30  # hard timeout 30 mins


# Reporter used in background submission manager, it is a dummy reporter that does nothing
class BackgroundSubmissionManagerReporter(MultiProgressReporter):
    def __init__(self):
        self.runs = []
    async def show(self, title: str):
        return
    def add_run(self, title: str) ->"BackgroundSubmissionManagerReporterRunProgressReporter":
        rep = BackgroundSubmissionManagerReporterRunProgressReporter(title)
        self.runs.append(rep)
        return rep
    def make_message(self):
        return
class BackgroundSubmissionManagerReporterRunProgressReporter(RunProgressReporter):
    async def _update_message(self):
        pass
    async def display_report(self, title: str, report: RunResultReport):
        pass


class BackgroundSubmissionManager:
    """
    This class manages submission in the backeground. It is responsible for
    submitting jobs to the backend, monitoring their progress, and updating the
    submissoin status in the database.

    It is also responsible for managing the workers, starting and stopping them
    as needed. By default, the api can maximum support 24 submission processes,

    Scale up: scale up up to max_workers based on the queue size and the
                number of running workers
    Scale down: each worker scale down automatically after hitting idle_seconds(hot)
                if there is no job in the queue.
    """

    def __init__(
        self,
        backend: KernelBackend,
        min_workers: int = 2,
        max_workers: int = 24,
        idle_seconds: int = 120,
    ):
        self.backend = backend
        self.queue: asyncio.Queue[JobItem] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []  # workers currently running
        self._live_tasks: set[
            asyncio.Task
        ] = set()  # tasks currently processing
        self.idle_seconds = (
            idle_seconds  # idle_seconds for each worker before scale down
        )
        # state variables

        self._state_lock = asyncio.Lock()
        self._accepting: bool = False
        self.min_workers = min_workers
        self.max_workers = max_workers

    async def start(self):
        logger.info("[Background Job] starting background submission manager")
        async with self._state_lock:
            self._accepting = True
            need = max(0, self.min_workers - len(self._workers))
        for _ in range(need):
            t = asyncio.create_task(self._worker_loop(), name="bg-worker")
            async with self._state_lock:
                self._workers.append(t)

    async def stop(self):
        logger.info(
            "[Background Job] stopping background submission manager..."
        )
        async with self._state_lock:
            self._accepting = False
            workers = list(self._workers)
            self._workers.clear()
        for t in workers:
            t.cancel()
        for t in workers:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        logger.info(
            "[Background Job] ...stopped all background submission manager"
        )

    async def enqueue(
        self,
        req: ProcessedSubmissionRequest,
        mode: SubmissionMode,
        sub_id: int,
    ) -> tuple[int, int]:
        async with self._state_lock:
            if not self._accepting:
                raise RuntimeError(
                    "[Background Job] Background Submission Manager is not"
                    "accepting new jobs right now"
                )
        logger.info("enqueueing submission %s", sub_id)
        now = dt.datetime.now(dt.timezone.utc)
        with self.backend.db as db:
            job_id = db.upsert_submission_job_status(
                sub_id,
                status="pending",
                last_heartbeat=now,
            )
        await self.queue.put(
            JobItem(job_id=job_id, sub_id=sub_id, req=req, mode=mode)
        )
        # if we have no workers and it does not hit maximum, start one
        await self._autoscale_up()
        return job_id, sub_id

    async def _worker_loop(self):
        """
        A worker will keep listening to the queue, and process the job in the queue.
        If the queue is empty, it will exit after idle_seconds.
        Each worker only handles one submission job at a time
        """
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(), timeout=self.idle_seconds
                    )
                    logger.info(
                        "[Background Job][worker %r] pick the submission job `%s`",
                        id(asyncio.current_task()),
                        item.sub_id,
                    )
                except asyncio.TimeoutError:
                    async with self._state_lock:
                        me = asyncio.current_task()
                        if (
                            len(self._workers) > self.min_workers
                            and me in self._workers
                        ):
                            try:
                                self._workers.remove(me)
                                logger.info(
                                    "[Background Job][worker %r] idle too long,"
                                    "scale down; existing workers=%d",
                                    me.get_name()
                                    if hasattr(me, "get_name")
                                    else id(me),
                                    len(self._workers),
                                )
                            except ValueError:
                                pass
                            return  # scale down: exit

                    continue

                t = asyncio.create_task(
                    self._run_one(item), name=f"submision-job-{item.sub_id}"
                )

                async with self._state_lock:
                    self._live_tasks.add(t)
                try:
                    await t  # wait submission job to finish
                finally:
                    logger.info(
                        "[Background Job][worker %r] finishes the submission job `%s`",
                        id(asyncio.current_task()),
                        item.sub_id,
                    )
                    async with self._state_lock:
                        self._live_tasks.discard(t)
                    self.queue.task_done()
        except asyncio.CancelledError:
            return

    async def _task_done_async(self, tt: asyncio.Task, item: JobItem):
        async with self._state_lock:
            self._live_tasks.discard(tt)
        self.queue.task_done()
        await self._autoscale_up()

    async def _run_one(self, item: JobItem):
        sub_id = item.sub_id
        now = dt.datetime.now(dt.timezone.utc)

        logger.info("[Background Job] start processing submission %s", sub_id)
        with self.backend.db as db:
            db.upsert_submission_job_status(
                sub_id, status="running", last_heartbeat=now
            )

        # heartbeat loop continuously update the last heartbeat time for the submission status
        stop_heartbeat = asyncio.Event()

        async def heartbeat():
            while not stop_heartbeat.is_set():
                await asyncio.sleep(HEARTBEAT_SEC)
                ts = dt.datetime.now(dt.timezone.utc)
                try:
                    with self.backend.db as db:
                        db.update_heartbeat_if_active(sub_id, ts)
                except Exception:
                    pass

        hb_task = asyncio.create_task(heartbeat(), name=f"hb-{sub_id}")
        try:
            reporter = BackgroundSubmissionManagerReporter()
            await asyncio.wait_for(
                self.backend.submit_full(
                    item.req, item.mode, reporter, sub_id
                ),
                timeout=HARD_TIMEOUT_SEC,
            )
            ts = dt.datetime.now(dt.timezone.utc)
            logger.info("[Background Job] submission %s succeeded", sub_id)
            with self.backend.db as db:
                db.upsert_submission_job_status(
                    sub_id, status="succeeded", last_heartbeat=ts
                )
        except asyncio.TimeoutError:
            ts = dt.datetime.now(dt.timezone.utc)
            with self.backend.db as db:
                db.upsert_submission_job_status(
                    sub_id,
                    status="timed_out",
                    last_heartbeat=ts,
                    error="hard timeout reached",
                )
        except Exception as e:
            ts = dt.datetime.now(dt.timezone.utc)
            logger.error(
                "[Background Job] submission job %s failed",
                sub_id,
                exc_info=True,
            )
            try:
                with self.backend.db as db:
                    db.upsert_submission_job_status(
                        sub_id,
                        status="failed",
                        last_heartbeat=ts,
                        error=str(e),
                    )
            except Exception:
                logger.error(
                    "[Background Job] Failed to write failed status for submission %s",
                    sub_id,
                )
        finally:
            stop_heartbeat.set()
            hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb_task

    async def _autoscale_up(self):
        async with self._state_lock:
            running = len(self._live_tasks)
            workers = len(self._workers)
            qsize = self.queue.qsize()

            desired = min(
                self.max_workers, max(self.min_workers, running + qsize)
            )
            need = desired - workers
            to_add = max(0, need)
            logger.info(
                "[Background Job] autoscale plan: add %d workers "
                "(max=%d, busy=%d, active=%s, enqueue=%d)",
                to_add,
                self.max_workers,
                running,
                workers,
                qsize,
            )
            for _ in range(to_add):
                logging.info(
                    "[Background Job] scale up: starting a new worker"
                )
                t = asyncio.create_task(self._worker_loop(), name="bg-worker")
                self._workers.append(t)
