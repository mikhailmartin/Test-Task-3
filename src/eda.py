import os

import joblib
import matplotlib.pyplot as plt

import my_ds_tools
import src

FIGURES_PATH = os.path.join('..', 'reports', 'figures')


def main(input_data_path: str) -> None:
    train_data = joblib.load(input_data_path).set_index('customer_ID')

    for num_feature in train_data.select_dtypes(include='float16').columns:
        try:
            fig, axes = my_ds_tools.eda.num_feature_report(
                data=train_data,
                feature_colname=num_feature,
                target_colname='target',
            )
            plt.savefig(os.path.join(FIGURES_PATH, f'{num_feature}.png'))
            plt.close()
        except:
            print(f'Не удалось для признака {num_feature}')

    for cat_feature in train_data.select_dtypes(include='category').columns:
        try:
            fig, axes = my_ds_tools.eda.cat_feature_report(
                data=train_data,
                feature_colname=cat_feature,
                target_colname='target',
            )
            plt.savefig(os.path.join(FIGURES_PATH, f'{cat_feature}.png'))
            plt.close()
        except:
            print(f'Не удалось для признака {cat_feature}')


if __name__ == '__main__':
    main(src.constants.TRAIN_DATA_PATH)
