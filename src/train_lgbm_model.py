import click
import joblib
import lightgbm

import src


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('output_model_path', type=click.Path())
def train_lgbm(input_data_path: str, output_model_path: str) -> None:
    train_data = joblib.load(input_data_path).set_index('customer_ID')
    X_train = train_data.drop(columns=['target', 'S_2'])
    y_train = train_data['target']

    model = lightgbm.LGBMClassifier(
        objective='binary',
        is_unbalance=True,
        boosting_type='gbdt',
        max_depth=5,
        num_leaves=26,
        random_state=src.constants.RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, output_model_path)


if __name__ == '__main__':
    train_lgbm()
