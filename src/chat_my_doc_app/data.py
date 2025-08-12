from pathlib import Path
from typing import Literal, Optional

import polars as pl
import typer


def load_imdb_data(
    list_file: str = "data/raw/IMDB_List.csv",
    reviews_file: str = "data/raw/IMDB_Reviews.csv"
) -> pl.DataFrame:
    """
    Load and join IMDB list and reviews data.

    Args:
        list_file: Path to the IMDB list CSV file
        reviews_file: Path to the IMDB reviews CSV file

    Returns:
        Combined dataframe with reviews joined to movie list
    """
    df_list = pl.read_csv(list_file)
    df_reviews = pl.read_csv(reviews_file)

    df = (
        df_reviews
        .join(
            df_list,
            left_on="imdb_id",
            right_on="id",
            how="left"
        )
    )

    return df

def transform_imdb_data(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .drop("")
        .rename({
            "review title": "review_title"
        })
    )

def save_to_parquet(
    df: pl.DataFrame,
    output_path: str,
    compression: Literal["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"] = "snappy"
) -> None:
    """
    Save dataframe to parquet format.

    Args:

        df: Polars dataframe to save
        output_path: Output file path
        compression: Compression type (snappy, gzip, lz4, zstd)
    """
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.write_parquet(output_path, compression=compression)


def load_csv(file_path: str, **kwargs) -> pl.DataFrame:
    """
    Load CSV file with polars.

    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pl.read_csv

    Returns:
        Polars dataframe
    """
    return pl.read_csv(file_path, **kwargs)

app = typer.Typer()


@app.command()
def process_imdb_data(
    list_file: str = typer.Option("data/raw/IMDB_List.csv", help="Path to IMDB list CSV file"),
    reviews_file: str = typer.Option("data/raw/IMDB_Reviews.csv", help="Path to IMDB reviews CSV file"),
    output_file: str = typer.Option("data/standard/imdb_reviews.parquet", help="Output parquet file path"),
    compression: str = typer.Option("snappy", help="Compression type for parquet file")
) -> None:
    """Process IMDB data: load, transform, and save to parquet format."""
    typer.echo(f"Loading data from {list_file} and {reviews_file}")

    df = load_imdb_data(list_file, reviews_file)
    df = transform_imdb_data(df)

    typer.echo(f"Saving processed data to {output_file}")
    save_to_parquet(df, output_file, compression)

    typer.echo(f"✅ Successfully processed {len(df)} records")


if __name__ == "__main__":
    app()
