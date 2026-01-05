from pathlib import Path


class PathUtils:
    """Utility class for handling paths."""

    DATAPATH = Path("./data")
    OUTPATH = Path("./out")

    @staticmethod
    def get_data_path() -> Path:
        """Get the data path.

        Returns:
            Path: The path to the data directory.

        """
        return PathUtils.DATAPATH

    @staticmethod
    def get_output_path() -> Path:
        """Get the output path.

        Returns:
            Path: The path to the output directory.

        """
        return PathUtils.OUTPATH
