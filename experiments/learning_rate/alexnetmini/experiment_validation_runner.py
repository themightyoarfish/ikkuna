from experiments.learning_rate.alexnetmini.experiment_validation import ex, schedules

for schedule in schedules:
    for _ in range(10):
        ex.run(config_updates={'schedule': schedule})
