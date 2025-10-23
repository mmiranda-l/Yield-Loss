import logging

def setup_logger(log_file='out.log', log_level=logging.DEBUG):
    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Create a file handler that logs messages to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create a console handler that logs messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger