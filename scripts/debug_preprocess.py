from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks, meta_dataset_collator
from torch.utils.data import DataLoader

def main():
    X,y = make_classification(n_samples=200, n_features=20, n_informative=10, random_state=42)
    X = X.astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
    clf = TabPFNClassifier(ignore_pretraining_limits=True, device='cpu', n_estimators=2, fit_mode='batched')
    clf._initialize_model_variables()

    split_fn = lambda X, y, stratify=None: train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    training_datasets = get_preprocessed_dataset_chunks(calling_instance=clf, X_raw=X_train, y_raw=y_train, split_fn=split_fn, max_data_size=None, model_type='classifier', equal_split_size=True, data_shuffle_seed=42, preprocessing_random_state=42, shuffle=True)

    loader = DataLoader(training_datasets, batch_size=1, collate_fn=meta_dataset_collator)
    for batch in loader:
        print('Batch repr:')
        print(type(batch))
        print('X_context type:', type(batch.X_context))
        print('len(X_context):', len(batch.X_context))
        for i, item in enumerate(batch.X_context):
            print(f'  X_context[{i}] type:', type(item))
            try:
                print(f'    shape:', getattr(item, 'shape', None))
            except Exception:
                pass
        print('y_context type:', type(batch.y_context), 'len:', len(batch.y_context))
        for i, item in enumerate(batch.y_context):
            print(f'  y_context[{i}] type:', type(item), 'shape:', getattr(item, 'shape', None))
        print('X_query type:', type(batch.X_query), 'len:', len(batch.X_query))
        for i, item in enumerate(batch.X_query):
            print(f'  X_query[{i}] type:', type(item), 'shape:', getattr(item, 'shape', None))
        print('y_query type:', type(batch.y_query), 'shape:', getattr(batch.y_query, 'shape', None))
        print('cat_indices type:', type(batch.cat_indices), 'len:', getattr(batch.cat_indices, '__len__', None) and len(batch.cat_indices))
        break

if __name__ == '__main__':
    main()
