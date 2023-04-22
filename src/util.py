from pathlib import Path
import pandas as pd


def xlsx_to_csv(input_dir: Path, output_dir: Path, sep: str = ','):
    """Converts all xlsx files in a directory to csv files.

    This function takes in an input directory containing xlsx files and converts them to csv files
    with the specified separator. The csv files are saved in the specified output directory.

    Args:
        input_dir (Path): The input directory containing xlsx files.
        output_dir (Path): The output directory where the csv files will be saved.
        sep (str): The separator to use in the csv file. Defaults to ','.

    """
    output_dir.mkdir(exist_ok=True)
    for file_path in input_dir.glob('*.xlsx'):
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f'Failed to read {file_path}, skipping. Exception: {e}')
            continue
        csv_file_path = output_dir / file_path.with_suffix('.csv').name
        try:
            df.to_csv(csv_file_path, index=False, sep=sep)
        except Exception as e:
            print(f'Failed to save {csv_file_path}, skipping. Exception: {e}')
            continue
        