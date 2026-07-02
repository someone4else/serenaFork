import logging

import pytest

from solidlsp.ls_utils import FileUtils


class TestIsBinaryFile:
    def test_plain_text_is_not_binary(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("hello world\n", encoding="utf-8")
        assert not FileUtils.is_binary_file(str(p))

    def test_nul_byte_marks_binary(self, tmp_path):
        p = tmp_path / "a.dll"
        p.write_bytes(b"MZ\x90\x00\x03\x00" + bytes(range(256)))
        assert FileUtils.is_binary_file(str(p))

    def test_utf16_bom_is_treated_as_text(self, tmp_path):
        # UTF-16 content contains NUL bytes but a BOM marks it as genuine text.
        p = tmp_path / "u.txt"
        p.write_bytes("﻿C:\\Users\\path".encode("utf-16-le"))
        assert not FileUtils.is_binary_file(str(p))

    def test_missing_file_is_not_binary(self, tmp_path):
        assert not FileUtils.is_binary_file(str(tmp_path / "nope.bin"))


class TestReadFile:
    def test_reads_utf8_without_logging(self, tmp_path, caplog):
        p = tmp_path / "a.txt"
        p.write_text("hello мир\n", encoding="utf-8")
        with caplog.at_level(logging.DEBUG, logger="solidlsp.ls_utils"):
            content = FileUtils.read_file(str(p), "utf-8")
        assert content == "hello мир\n"
        assert caplog.records == []

    def test_binary_raises_without_error_logging(self, tmp_path, caplog):
        p = tmp_path / "a.dll"
        p.write_bytes(b"MZ\x90\x00" + bytes(range(256)) + b"\x00tail")
        with caplog.at_level(logging.DEBUG, logger="solidlsp.ls_utils"):
            with pytest.raises(UnicodeDecodeError):
                FileUtils.read_file(str(p), "utf-8")
        # Binary read failures are expected and must not be logged at WARNING/ERROR.
        assert all(r.levelno < logging.WARNING for r in caplog.records)

    def test_legacy_encoding_is_detected_and_logged_once(self, tmp_path, caplog):
        p = tmp_path / "c.txt"
        p.write_bytes("Привет и добро пожаловать".encode("cp1251"))
        with caplog.at_level(logging.DEBUG, logger="solidlsp.ls_utils"):
            first = FileUtils.read_file(str(p), "utf-8")
            info_after_first = [r for r in caplog.records if r.levelno == logging.INFO]
            caplog.clear()
            second = FileUtils.read_file(str(p), "utf-8")
        assert "Привет" in first
        assert "Привет" in second
        # Detection is logged at most once (INFO); subsequent cached reads log nothing.
        assert len(info_after_first) <= 1
        assert all(r.levelno < logging.WARNING for r in caplog.records)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FileUtils.read_file(str(tmp_path / "nope.txt"), "utf-8")
