import json
import logging

from preprocess import Preprocess




if __name__ == "__main__":

    level = logging.DEBUG



    logger = logging.getLogger()
    console_handler = logging.StreamHandler()

    logger.setLevel(level)
    console_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    with open('params.json', 'r') as file:
        params = json.load(file)

    pp = Preprocess(logger=logger ,path=params["data_path"], text_column=params["text_column"], label_column=params["label_column"])

    pp.run()
    print('e')