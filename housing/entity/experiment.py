class Experiment:
    running_status = False
    def __new__(cls, *args, **kwargs):
        if Experiment.running_status:
            raise Exception("Experiment is already running hence can't be instantiated new one.")
        return super(Experiment, cls).__new__(cls, *args, **kwargs)
    
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.running_status = Experiment.running_status