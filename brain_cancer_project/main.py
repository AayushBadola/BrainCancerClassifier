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
    parser.add_argument('--eval_on_val', action=argparse.BooleanOptionalAction, default=True,
                        help="Evaluate on validation set for 'evaluate' or 'full_run'.")
    parser.add_argument('--eval_on_test_dir', type=str, default=None,
                        help="Path to separate test dataset for 'evaluate' or 'full_run'.")
    parser.add_argument('--image_path', type=str, help="Path to single image for 'predict'.")
    parser.add_argument('--sample_predict', action='store_true',
                        help="Predict on random sample from dataset if 'predict' and no --image_path.")
    parser.add_argument('--sample_class', type=str, default=None,
                        help="Specific class for --sample_predict.")

    args = parser.parse_args()

    print(f"Selected action: {args.action}")
    utils.set_seeds()

    if args.action == 'test_data_load':
        print("\n--- Initiating Data Loading Test ---")
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
        if not args.eval_on_val and not args.eval_on_test_dir:
            print("Evaluation: specify --eval_on_val (default) or --eval_on_test_dir.")
            parser.print_help(); return
        evaluator.evaluate_model(model_path=args.model_path, 
                                 on_validation_data=args.eval_on_val, 
                                 on_test_data_dir=args.eval_on_test_dir)
    
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
        print("\n--- Initiating Full Run (Train then Evaluate) ---")
        print("\nStep 1: Training Model")
        trained_model_obj, _ = trainer.train_model()
        if trained_model_obj:
            print("\nStep 2: Evaluating Model")
            evaluator.evaluate_model(model_path=config.MODEL_PATH,
                                     on_validation_data=args.eval_on_val,
                                     on_test_data_dir=args.eval_on_test_dir)
        else:
            print("Training failed, skipping evaluation.")
            
    else:
        print(f"Unknown action: {args.action}"); parser.print_help()

if __name__ == '__main__':
    if not os.path.exists(config.DATASET_DIR) or not os.listdir(config.DATASET_DIR):
        print(f"CRITICAL Error: Dataset directory '{config.DATASET_DIR}' is empty or does not exist.")
        print("Please ensure your 'Brain_Cancer' dataset is correctly placed and populated, e.g.:")
        print(f"{config.BASE_DIR}/")
        print("├── Brain_Cancer/")
        print("│   ├── brain_glioma/ (with images)")
        print("│   ├── brain_menin/ (with images)")
        print("│   └── brain_tumor/ (with images)")
        print("...")
        sys.exit(1)
    
    main()
