import tempfile
import unittest
from pathlib import Path

from src.agent.intent_classifier import IntentType
from src.agent.session_store import InMemorySessionStore, SQLiteSessionStore


class TestSQLiteSessionStore(unittest.TestCase):
    def test_persists_conversations_messages_and_procedure_state_across_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nested" / "chat.db")
            store1 = SQLiteSessionStore(db_path=db_path)

            session = store1.get_or_create("conv-1")
            session.push_intent(IntentType.PROCEDURAL)
            session.active_process = "कम्पनी दर्ता"
            session.current_step = 2
            session.total_steps = 5
            session.add_message("user", "कम्पनी दर्ता कसरी गर्ने र के के कागजात चाहिन्छ?")
            session.add_message(
                "assistant",
                "कम्पनी दर्ता गर्न दफा ३(१) अनुसार आवेदन दिनुपर्छ।",
                citations=[{"label": "कम्पनी ऐन, २०६३, दफा ३(१)", "text": "पाठ"}],
                steps=["Intent: procedural", "Generating answer…"],
            )
            store1.save(session)
            store1.close()

            store2 = SQLiteSessionStore(db_path=db_path)
            loaded = store2.get("conv-1")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(loaded.title.startswith("कम्पनी दर्ता"))
            self.assertEqual(loaded.active_process, "कम्पनी दर्ता")
            self.assertEqual(loaded.current_step, 2)
            self.assertEqual(loaded.intent_history, [IntentType.PROCEDURAL])
            self.assertEqual(len(loaded.messages), 2)
            self.assertEqual(loaded.messages[0].role, "user")
            self.assertEqual(loaded.messages[1].citations[0]["label"], "कम्पनी ऐन, २०६३, दफा ३(१)")
            self.assertEqual(loaded.messages[1].steps, ["Intent: procedural", "Generating answer…"])

            listed = store2.list_sessions()
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["session_id"], "conv-1")
            self.assertEqual(listed[0]["message_count"], 2)

            self.assertTrue(store2.rename("conv-1", "Company Registration"))
            self.assertEqual(store2.get("conv-1").title, "Company Registration")  # type: ignore[union-attr]

            self.assertTrue(store2.delete("conv-1"))
            self.assertIsNone(store2.get("conv-1"))
            self.assertEqual(len(store2), 0)
            store2.close()

    def test_replace_last_assistant_message_avoids_duplicate_user_turns(self) -> None:
        store = InMemorySessionStore()
        session = store.get_or_create("conv-regen")
        session.add_message("user", "दफा ३ के भन्छ?")
        session.add_message("assistant", "पुरानो जवाफ")
        session.replace_last_assistant_message("नयाँ जवाफ", citations=[{"label": "दफा ३"}])

        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[0].content, "दफा ३ के भन्छ?")
        self.assertEqual(session.messages[1].content, "नयाँ जवाफ")
        self.assertEqual(session.messages[1].citations, [{"label": "दफा ३"}])


if __name__ == "__main__":
    unittest.main()
