import unittest

from scripts.reorder_project_log import parse


class ReorderProjectLogParserTest(unittest.TestCase):
    def test_trailing_html_comment_is_preserved_as_footer(self):
        text = (
            "# Log\n\n"
            "## Current state\ncurrent\n\n"
            "## 2026-07-22 — result\nresult\n\n"
            "<!-- footer -->\n"
        )

        _, sections, footer = parse(text)

        self.assertEqual(len(sections), 2)
        self.assertEqual(footer, "<!-- footer -->\n")

    def test_mid_document_html_comment_fails_closed(self):
        text = (
            "# Log\n\n"
            "## 2026-07-22 — first\nfirst\n\n"
            "<!-- misplaced footer -->\n\n"
            "## 2026-07-21 — hidden\nhidden\n"
        )

        with self.assertRaisesRegex(SystemExit, "not trailing"):
            parse(text)

    def test_unterminated_html_comment_fails_closed(self):
        text = "# Log\n\n## 2026-07-22 — result\nresult\n\n<!-- footer\n"

        with self.assertRaisesRegex(SystemExit, "unterminated"):
            parse(text)


if __name__ == "__main__":
    unittest.main()
