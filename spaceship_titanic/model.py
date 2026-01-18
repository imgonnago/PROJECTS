from flaml import AutoML

def model():
    automl = AutoML()

    settings = {
        "time_budget": 60,
        "metric": 'accuracy',
        "task":'classification',
        "seed":42
    }
    return automl, settings

