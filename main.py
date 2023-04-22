from pathlib import Path
import pandas as pd


def xlsx_to_csv(input_dir: Path, output_dir: Path, sep: str = ','):
    output_dir.mkdir(exist_ok=True)
    for file_path in input_dir.glob('*.xlsx'):
        df = pd.read_excel(file_path)
        csv_file_path = output_dir / file_path.with_suffix('.csv').name
        df.to_csv(csv_file_path, index=False, sep=sep)

if __name__=='__main__':

    # # convert xlsx to csv file
    xlsx_dir = Path('./data/xlsx')
    csv_dir = Path('./data/csv')
    # xlsx_to_csv(xlsx_dir, csv_dir, sep='||')

    # read csv files from csv_dir
    
