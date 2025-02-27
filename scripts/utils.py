class DataFromJSON():
    """
    Class to manage the data of the model.
    """
    def __init__(self, json_dict, data_type: str):
        self.__class_name = "DataFromJSON"
        self.__data_type = data_type
        self.loop(json_dict)

    def __str__(self):
        return f"{self.__class_name} object with data type: {self.__data_type}"

    def loop(self, json_dict):
        if not isinstance(json_dict, dict):
            return
        for key, value in json_dict.items():
            if isinstance(value, dict):
                self.loop(value)
            else:
                if hasattr(self, key):
                    raise ValueError(f"Variable {key} already exists in the class. Rename the json key in your configuration file.")
                else:
                    setattr(self, key, value)