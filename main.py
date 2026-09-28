
from core.train.train import ModelsTraining
from use_cases.carla_driving.carla_models import AutonomousDriving
from use_cases.carla_driving.carla_settings import CarlaProjectConfig, CONFIG
from use_cases.carla_driving.carla_provider import CarlaDataProvider
from use_cases.carla_driving.carla_nas import CarlaNASProblem
from core.nas.base_nas import NASHistoryCallback
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2

from use_cases.carla_driving.export_engine import ExportEngine


if __name__ == "__main__":

   config = CONFIG
   data_provider = CarlaDataProvider(config=config)
   train_loader, val_loader, test_loader = data_provider.prepare_datasets()

   problem = CarlaNASProblem(
      train_loader=train_loader,
      validation_loader=val_loader,
      test_loader=test_loader,
      config=config,
      trainer_cls=ModelsTraining,
      model_cls=AutonomousDriving
   )

   algorithm = NSGA2(
      pop_size=config.nas.population_size
   )

   print("[Test] Setting up History Callback...")
   callback = NASHistoryCallback(
      output_path="use_cases/carla_driving/exported_models/test_nas_history.json",
      metric_names=["val_weighted_mae", "tinyml_cost"]
   )

   print("[Test] Starting optimization dry run (2 generations)...")
   try:
      res = minimize(
         problem,
         algorithm,
         termination=("n_gen", 2),
         callback=callback,
         seed=42,
         verbose=True
      )
      print("\n[Test Success] NAS optimization run completed successfully!")
      if res.X is not None:
         print(f"Best solutions found: {len(res.X)}")

      exporter = ExportEngine(
        config=config,
        export_dir=config.ai.exported_models_path,
        trainer_cls=ModelsTraining
      )

      top_profiles = exporter.export_top_models(
        res_X=res.X,
        res_F=res.F,
        train_loader=train_loader,
        validation_loader=val_loader,
        test_loader=test_loader,
        top_n=5
      )

   except Exception as error:
      print(f"\n[Test Error] Pipeline execution failed: {error}")
      import traceback
      traceback.print_exc()