"""
Restructure invite codes to support multiple leaderboards per code.
Moves from leaderboard_invite.leaderboard_id to a junction table leaderboard_invite_scope.
"""

from yoyo import step

__depends__ = {'20260311_01_closed-leaderboard-visibility'}


steps = [
    # 1. Create the junction table
    step(
        """
        CREATE TABLE leaderboard.leaderboard_invite_scope (
            invite_id INTEGER NOT NULL REFERENCES leaderboard.leaderboard_invite(id) ON DELETE CASCADE,
            leaderboard_id INTEGER NOT NULL REFERENCES leaderboard.leaderboard(id) ON DELETE CASCADE,
            PRIMARY KEY (invite_id, leaderboard_id)
        );
        """,
        """
        DROP TABLE leaderboard.leaderboard_invite_scope;
        """
    ),
    # 2. Migrate existing data from leaderboard_invite.leaderboard_id into the junction table
    step(
        """
        INSERT INTO leaderboard.leaderboard_invite_scope (invite_id, leaderboard_id)
        SELECT id, leaderboard_id FROM leaderboard.leaderboard_invite
        WHERE leaderboard_id IS NOT NULL;
        """,
        """
        UPDATE leaderboard.leaderboard_invite li
        SET leaderboard_id = lis.leaderboard_id
        FROM leaderboard.leaderboard_invite_scope lis
        WHERE li.id = lis.invite_id;
        """
    ),
    # 3. Drop the old column and index
    step(
        """
        DROP INDEX IF EXISTS leaderboard.idx_leaderboard_invite_leaderboard;
        ALTER TABLE leaderboard.leaderboard_invite DROP COLUMN leaderboard_id;
        """,
        """
        ALTER TABLE leaderboard.leaderboard_invite
            ADD COLUMN leaderboard_id INTEGER REFERENCES leaderboard.leaderboard(id) ON DELETE CASCADE;
        UPDATE leaderboard.leaderboard_invite li
        SET leaderboard_id = lis.leaderboard_id
        FROM leaderboard.leaderboard_invite_scope lis
        WHERE li.id = lis.invite_id;
        CREATE INDEX idx_leaderboard_invite_leaderboard
            ON leaderboard.leaderboard_invite (leaderboard_id);
        """
    ),
    # 4. Index on the junction table
    step(
        """
        CREATE INDEX idx_leaderboard_invite_scope_leaderboard
        ON leaderboard.leaderboard_invite_scope (leaderboard_id);
        """,
        """
        DROP INDEX IF EXISTS leaderboard.idx_leaderboard_invite_scope_leaderboard;
        """
    ),
]
