import http.server
import threading
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llmebench.datasets.dataset_base import DatasetBase


class ArchiveHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, directory="tests/datasets/archives")


class SignalingHTTPServer(http.server.HTTPServer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ready_event = threading.Event()

    def service_actions(self):
        self.ready_event.set()


class MockDataset(DatasetBase):
    def metadata():
        return {}

    def get_data_sample():
        return {"input": "input", "label": "label"}

    def load_data(self, data_path):
        return [self.get_data_sample() for _ in range(100)]


class TestDatasetAutoDownload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = SignalingHTTPServer(("", 0), ArchiveHandler)
        cls.port = cls.httpd.server_address[1]

        cls.test_server = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.test_server.start()
        cls.httpd.ready_event.wait()

    @classmethod
    def tearDownClass(cls):
        if cls.httpd:
            cls.httpd.shutdown()
            cls.httpd.server_close()
        cls.test_server.join()

    def check_downloaded(self, data_dir, dataset_name, extension):
        downloaded_files = list(data_dir.iterdir())
        downloaded_filenames = [f.name for f in downloaded_files if f.is_file()]
        self.assertEqual(len(downloaded_files), 2)
        self.assertIn(f"{dataset_name}.{extension}", downloaded_filenames)

        extracted_directories = [d for d in downloaded_files if d.is_dir()]
        extracted_directory_names = [d.name for d in extracted_directories]
        self.assertIn(f"{dataset_name}", extracted_directory_names)
        self.assertEqual(len(extracted_directory_names), 1)

        dataset_files = [f.name for f in extracted_directories[0].iterdir()]
        self.assertIn("train.txt", dataset_files)
        self.assertIn("test.txt", dataset_files)

    def test_auto_download_zip(self):
        "Test automatic downloading and extraction of *.zip datasets"

        data_dir = TemporaryDirectory()
        data_dir_path = Path(data_dir.name)

        dataset = MockDataset(data_dir=data_dir_path)
        self.assertTrue(
            dataset.download_dataset(
                data_dir=data_dir.name,
                download_url=f"http://localhost:{self.port}/Mock.zip",
            )
        )

        self.check_downloaded(data_dir_path, "Mock", "zip")

    def test_auto_download_tar(self):
        "Test automatic downloading and extraction of *.tar datasets"

        data_dir = TemporaryDirectory()
        data_dir_path = Path(data_dir.name)

        dataset = MockDataset(data_dir=data_dir_path)
        self.assertTrue(
            dataset.download_dataset(
                data_dir=data_dir.name,
                download_url=f"http://localhost:{self.port}/Mock.tar",
            )
        )

        self.check_downloaded(data_dir_path, "Mock", "tar")

    def test_auto_download_tar_gz(self):
        "Test automatic downloading and extraction of *.tar.gz datasets"

        data_dir = TemporaryDirectory()
        data_dir_path = Path(data_dir.name)

        dataset = MockDataset(data_dir=data_dir_path)
        self.assertTrue(
            dataset.download_dataset(
                data_dir=data_dir.name,
                download_url=f"http://localhost:{self.port}/Mock.tar.gz",
            )
        )

        self.check_downloaded(data_dir_path, "Mock", "tar.gz")

    def test_auto_download_tar_bz2(self):
        "Test automatic downloading and extraction of *.tar.bz2 datasets"

        data_dir = TemporaryDirectory()
        data_dir_path = Path(data_dir.name)

        dataset = MockDataset(data_dir=data_dir_path)
        self.assertTrue(
            dataset.download_dataset(
                data_dir=data_dir.name,
                download_url=f"http://localhost:{self.port}/Mock.tar.bz2",
            )
        )

        self.check_downloaded(data_dir_path, "Mock", "tar.bz2")

    def test_auto_download_tar_xz(self):
        "Test automatic downloading and extraction of *.tar.xz datasets"

        data_dir = TemporaryDirectory()
        data_dir_path = Path(data_dir.name)

        dataset = MockDataset(data_dir=data_dir_path)
        self.assertTrue(
            dataset.download_dataset(
                data_dir=data_dir.name,
                download_url=f"http://localhost:{self.port}/Mock.tar.xz",
            )
        )

        self.check_downloaded(data_dir_path, "Mock", "tar.xz")

    def test_auto_download_default_url(self):
        "Test automatic downloading when download url is not provided"

        data_dir = TemporaryDirectory()
        data_dir_path = Path(data_dir.name)

        dataset = MockDataset(data_dir=data_dir_path)
        self.assertTrue(
            dataset.download_dataset(
                data_dir=data_dir.name, default_url=f"http://localhost:{self.port}/"
            )
        )

        self.check_downloaded(data_dir_path, "Mock", "zip")

    @patch.dict(
        "os.environ",
        {
            "DEFAULT_DOWNLOAD_URL": "http://invalid.llmebench-server.org",
        },
    )
    def test_auto_download_metadata_url(self):
        "Test automatic downloading when download url is provided in metadata"

        data_dir = TemporaryDirectory()
        data_dir_path = Path(data_dir.name)

        class MockDatasetWithDownloadURL(MockDataset):
            def metadata():
                return {"download_url": f"http://localhost:{self.port}/Mock.zip"}

        dataset = MockDatasetWithDownloadURL(data_dir=data_dir_path)
        self.assertTrue(dataset.download_dataset(data_dir=data_dir.name))

        self.check_downloaded(data_dir_path, "MockDatasetWithDownloadURL", "zip")

    @patch.dict(
        "os.environ",
        {
            "DEFAULT_DOWNLOAD_URL": "http://invalid.llmebench-server.org",
        },
    )
    def test_auto_download_non_existent(self):
        "Test automatic downloading when dataset is not actually available"

        data_dir = TemporaryDirectory()
        data_dir_path = Path(data_dir.name)

        class MockDatasetWithDownloadURL(MockDataset):
            def metadata():
                return {
                    "download_url": f"http://localhost:{self.port}/InvalidDataset.zip"
                }

        dataset = MockDatasetWithDownloadURL(data_dir=data_dir_path)
        self.assertFalse(
            dataset.download_dataset(
                data_dir=data_dir.name,
                download_url="http://invalid.llmebench-server.org/Dataset.zip",
            )
        )


class TestDatasetCaching(unittest.TestCase):
    def test_cache_existing_file(self):
        "Test if an existing file _does not_ trigger a download"

        data_dir = TemporaryDirectory()
        data_dir_path = Path(data_dir.name)

        # Copy a archive to the download location
        archive_file = Path("tests/datasets/archives/Mock.zip")
        copy_archive_file = data_dir_path / "Mock.zip"
        copy_archive_file.write_bytes(archive_file.read_bytes())

        # download_dataset should not reach out to the invalid server,
        # since file is present locally
        dataset = MockDataset(data_dir=data_dir_path)
        self.assertTrue(
            dataset.download_dataset(
                data_dir=data_dir.name,
                download_url="http://invalid.llmebench-server.org/ExistingData.zip",
            )
        )
