import json
import os

from libkernelbot.leaderboard_db import LeaderboardDB
from libkernelbot.utils import setup_logging

logger = setup_logging(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
AUDIT_MODEL = "openai/gpt-4o-mini"

AUDIT_SYSTEM = """You are a code auditor for a GPU kernel programming competition.

You will be given reference/evaluation code and a user's submitted solution.

Your job is to determine if the submission is cheating. Common cheating patterns:
- Hardcoding expected outputs instead of computing them
- Bypassing the evaluation/correctness checks
- Exploiting the test harness rather than solving the problem
- Returning precomputed values without doing real computation
- Patching or monkey-patching evaluation functions

Respond with ONLY a JSON object (no markdown, no extra text):
{"is_cheating": true/false, "explanation": "brief explanation"}"""


async def audit_submission(submission_id: int, db: LeaderboardDB) -> dict | None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.debug("OPENROUTER_API_KEY not set, skipping audit for submission %s", submission_id)
        return None

    try:
        import openai

        with db:
            submission = db.get_submission_by_id(submission_id)
            if submission is None:
                logger.warning("Submission %s not found for audit", submission_id)
                return None

            task_json = db.get_leaderboard_task_by_id(submission["leaderboard_id"])

        if not task_json:
            logger.warning("No task found for leaderboard %s", submission["leaderboard_id"])
            return None

        # Extract reference code from the task files
        reference_code = ""
        if isinstance(task_json, dict) and "files" in task_json:
            for filename, content in task_json["files"].items():
                reference_code += f"--- {filename} ---\n{content}\n\n"

        if not reference_code:
            reference_code = json.dumps(task_json, indent=2)

        submission_code = submission["code"]
        user_msg = (
            "Reference/evaluation code:\n```\n"
            + reference_code
            + "\n```\n\nSubmitted code:\n```\n"
            + submission_code
            + "\n```"
        )

        client = openai.AsyncOpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        response = await client.chat.completions.create(
            model=AUDIT_MODEL,
            messages=[
                {"role": "system", "content": AUDIT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=512,
        )

        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)

        is_cheating = bool(result.get("is_cheating", False))
        explanation = result.get("explanation", "")

        with db:
            db.create_submission_audit(submission_id, is_cheating, explanation, AUDIT_MODEL)

        logger.info(
            "Audit for submission %s: is_cheating=%s", submission_id, is_cheating
        )
        return {"is_cheating": is_cheating, "explanation": explanation, "model": AUDIT_MODEL}

    except Exception:
        logger.exception("Failed to audit submission %s", submission_id)
        return None
