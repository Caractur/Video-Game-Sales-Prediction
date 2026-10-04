# Video Game Sales Prediction

Can you predict how many copies a game will sell worldwide before it launches? This notebook tries, finds out the answer is mostly no, and works out why.

It was built by a five-person team in a one-day machine-learning hackathon during a data science bootcamp. The brief: pick a dataset, clean it, explore it, and compare at least two models.

## Data

The [Video Game Sales dataset on Kaggle](https://www.kaggle.com/datasets/gregorut/videogamesales) lists 16,598 games that sold over 100,000 copies, with platform, release year, genre, publisher, and sales in millions for North America, Europe, Japan, the rest of the world and the global total.

It isn't included here. Download `vgsales.csv` from Kaggle and put it in `data/`.

## What the notebook does

**Cleaning.** No duplicate rows. The only gaps are in `Year` and `Publisher`, about 1.85% of rows, so those rows are dropped (16,598 → 16,291). `Rank` and `Name` are dropped as features: rank is just global sales re-ordered, and the name is effectively an ID a model could memorise.

**Attempt 1: pre-launch details only.** A random forest predicts `Global_Sales` from platform, year, genre and publisher, the only things a studio knows before release. It scores **R² = 0.03** on the test set, which means it explains almost none of the variation.

**Why it failed.** Sales follow an extreme long tail: the average game sells about 540,000 copies, while a few blockbusters sell tens of millions. Grouping platforms by manufacturer and plotting releases per year shows the late-2000s flood of titles, led by Nintendo (the DS has more games than any other platform), which piles up thousands of low sellers with the same metadata as the occasional hit. Platform, genre and publisher can't separate the two.

**Attempt 2: adding regional sales.** With regional figures included, things change sharply. A single-feature linear regression per region shows how strongly each market tracks the global total:

| Region | R² (fit) | R² (5-fold CV) | MAE (CV) |
|---|---|---|---|
| North America | 0.886 | 0.871 | 0.202 |
| Europe | 0.816 | 0.805 | 0.246 |
| Other | 0.559 | 0.484 | 0.325 |
| Japan | 0.376 | 0.374 | 0.505 |

North America dominates because it has the largest sales in almost every row. A random forest on regional sales plus metadata, trained on log-transformed sales, reaches **test R² = 0.81** (MAE 0.045 million).

## The honest caveat

`Global_Sales` is essentially the sum of the four regional columns, so the second set of models is learning to rebuild a total from its parts rather than forecasting anything. Attempt 1 is the realistic business question, and it shows that this dataset alone can't answer it. Getting further would take signals that exist before launch, like franchise history, marketing spend, pre-orders or review scores.

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook video_game_sales.ipynb
```

Run from the repository root with `data/vgsales.csv` in place. The notebook saves its cleaned copy as `data/vgsales_clean.csv`.
