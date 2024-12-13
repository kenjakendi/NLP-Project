


class Data:
    def __init__(self, logger: Logger,  path: str, text_column: str, label_column: str):
        self.logger = logger
        self.logger.info("Loading data STARTED ...")
        self.data = self.load_data(path, text_column, label_column)
        self.logger.info("Loading data FINISHED.")