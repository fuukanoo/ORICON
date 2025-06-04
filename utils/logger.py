import logging

def init_logger(name='ORICON', level=logging.INFO):
    global _logger_initialized
    _logger_initialized = False
    
    if _logger_initialized:
        return logging.getLogger(name)
    
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    _logger_initialized = True
    logger.info(f"Logger '{name}' initialized successfully")
    
    return logger

def get_logger(name='ORICON'):
    logger = logging.getLogger(name)
    
    # If logger has no handlers, it means it hasn't been properly initialized
    if not logger.hasHandlers() and not _logger_initialized:
        return init_logger(name)
    
    return logger