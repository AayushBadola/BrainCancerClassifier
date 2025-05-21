import argparse
import os
import sys

project_root_abs = os.path.dirname(os.path.abspath(__file__))
if project_root_abs not in sys.path:
    sys.path.insert(0, project_root_abs)
parent_of_project_root = os.path.dirname(project_root_abs)
if parent_of_project_root not in sys.path:
    sys.path.insert(0, parent_of_project_root)

try:
    from . import train as trainer
    from . import evaluate as evaluator
    from . import predict as predictor
    from . import config
    from . import utils
    from . import data_loader
except ImportError:
    import train as trainer
    import evaluate as evaluator
    import predict as predictor
    import config
    import utils
    import data_loader

def main():
    parser = argparse.ArgumentParser(description="Brain Cancer MRI Classification Project")
    parser.add_argument('action', choices=['train', 'evaluate', 'predict', 'full_run', 'test_data_load'],
                        help="Action: 'train', 'evaluate', 'predict', 'full_run', 'test_data_load'.")
    
    parser.add_argument('--model_path', type=str, default=config.MODEL_PATH,
                        help="Path to model for evaluation/prediction.")
    parser.add_argument('--eval_on_val', action='store_true', default=False,
                        help="Evaluate on validation set (derived from training data).")
    parser.add_argument('--eval_on_test_dir', type=str, default=None,
                        help="Path to separate test dataset. Overrides default test set from config.")
    parser.add_argument('--image_path', type=str, help="Path to single image for 'predict'.")
    parser.add_argument('--sample_predict', action='store_true',
                        help="Predict on random sample from training dataset if 'predict' and no --image_path.")
    parser.add_argument('--sample_class', type=str, default=None,
                        help="Specific class for --sample_predict.")

    args = parser.parse_args()

    print(f"Selected action: {args.action}")
    utils.set_seeds()

    if args.action == 'test_data_load':
        print("\n--- Initiating Data Loading Test (from training data dir) ---")
        train_ds, val_ds, cls_names, num_cls = data_loader.load_datasets()
        if train_ds and cls_names:
            print(f"Data loading test successful. Found {num_cls} classes: {cls_names}")
            data_loader.visualize_sample_data(train_ds, cls_names)
        else:
            print("Data loading test failed.")
        return

    if args.action == 'train':
        print("\n--- Initiating Training ---")
        trainer.train_model()
    
    elif args.action == 'evaluate':
        print("\n--- Initiating Evaluation ---")
        
        cli_test_dir = args.eval_on_test_dir
        use_validation = args.eval_on_val

        if cli_test_dir:
            print(f"Evaluating on specified test directory: {cli_test_dir}")
            evaluator.evaluate_model(model_path=args.model_path, 
                                     on_validation_data=False, 
                                     on_test_data_dir=cli_test_dir)
        elif use_validation:
            print("Evaluating on validation set (derived from training data).")
            evaluator.evaluate_model(model_path=args.model_path, 
                                     on_validation_data=True, 
                                     on_test_data_dir=None)
        elif config.TEST_DATA_DIR and os.path.exists(config.TEST_DATA_DIR):
            print(f"Defaulting evaluation to dedicated test set from config: {config.TEST_DATA_DIR}")
            evaluator.evaluate_model(model_path=args.model_path,
                                     on_validation_data=False,
                                     on_test_data_dir=config.TEST_DATA_DIR)
        else:
            print("Warning: No dedicated test set found in config or specified via CLI, and not evaluating on validation set.")
            print("Please specify --eval_on_val or ensure config.TEST_DATA_DIR is valid.")
            parser.print_help()
            return
            
    elif args.action == 'predict':
        print("\n--- Initiating Prediction ---")
        if args.image_path:
            if not os.path.exists(args.image_path):
                print(f"Error: Image path for prediction does not exist: {args.image_path}"); return
            predictor.predict_single_image(args.image_path)
        elif args.sample_predict:
            predictor.predict_on_sample_from_dataset(class_to_sample=args.sample_class)
        else:
            print("Predict: provide --image_path or use --sample_predict."); parser.print_help()

    elif args.action == 'full_run':
        print("\n--- Initiating Full Run (Train then Evaluate on Test Set) ---")
        print("\nStep 1: Training Model")
        trained_model_obj, _ = trainer.train_model()
        if trained_model_obj:
            print("\nStep 2: Evaluating Model (on dedicated test set if available)")
            evaluation_test_dir = args.eval_on_test_dir
            if not evaluation_test_dir and config.TEST_DATA_DIR and os.path.exists(config.TEST_DATA_DIR):
                evaluation_test_dir = config.TEST_DATA_DIR
                print(f"Using dedicated test set for full_run evaluation: {evaluation_test_dir}")
            elif not evaluation_test_dir and args.eval_on_val:
                 print("Full_run: Evaluating on validation set as no test set specified/found.")
                 evaluator.evaluate_model(model_path=config.MODEL_PATH,
                                     on_validation_data=True,
                                     on_test_data_dir=None)
                 return
            elif not evaluation_test_dir:
                print("Full_run: No test set for evaluation specified or found in config. Skipping test evaluation.")
                return

            evaluator.evaluate_model(model_path=config.MODEL_PATH,
                                     on_validation_data=False,
                                     on_test_data_dir=evaluation_test_dir)
        else:
            print("Training failed, skipping evaluation.")
            
    else:
        print(f"Unknown action: {args.action}"); parser.print_help()

if __name__ == '__main__':
    if not os.path.exists(config.DATASET_DIR) or not os.listdir(config.DATASET_DIR):
        print(f"CRITICAL Error: Training dataset directory '{config.DATASET_DIR}' is empty or does not exist.")
        print("Please ensure your training data ('brain_cancer/' folder by default) is correctly placed and populated.")
        sys.exit(1)
    
    if not os.path.exists(config.TEST_DATA_DIR) or not os.listdir(config.TEST_DATA_DIR):
        print(f"Warning: Dedicated test dataset directory '{config.TEST_DATA_DIR}' is empty or does not exist.")
        print("The 'evaluate' action might default to validation set or fail if it expects this.")

    main()

