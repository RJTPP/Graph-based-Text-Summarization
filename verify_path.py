# For testing absolute path

import setting

if __name__ == "__main__":
    print(f"Based directory          : {setting.BASED_DIR}")
    print(f"Data directory           : {setting.DATA_DIR}")
    print(f"Cached directory         : {setting.CACHE_DIR}")
    print(f"Nltk path                : {setting.NLTK_PATH}")
    print(f"Output directory         : {setting.OUTPUT_DIR}")
    print(f"Validation file          : {setting.VALIDATION_FILE}")