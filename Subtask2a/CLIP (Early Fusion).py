import json
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from transformers import BertTokenizer, BertModel
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

class DataGenerator(Sequence):
    def __init__(self, data, batch_size, bert_model, bert_tokenizer, clip_processor, clip_model, data_type, label_binarizer, max_length=77, is_training=True, labeled=True, image_dir=None):
        self.data = data
        self.batch_size = batch_size
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.label_binarizer = label_binarizer
        self.max_length = max_length
        self.indices = np.arange(len(self.data))
        self.is_training = is_training
        self.labeled = labeled
        self.image_dir = data_paths[data_type]['image_dir']  # Get image directory based on data_type

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.data[k] for k in batch_indices]
        
        if self.labeled:
            X, y = self.preprocess_data(batch)
            if self.is_training:
                return X, y
            else:
                batch_ids = [sample["id"] for sample in batch]
                return X, y, batch_ids
        else:
            X = self.preprocess_data(batch, labeled=False)
            if not self.is_training:
                batch_ids = [sample["id"] for sample in batch]
            return X, batch_ids 

        
    def load_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                return img.convert('RGB')
        except IOError:
            print(f"Error in loading image: {full_path}. Using a placeholder image.")
            return Image.new('RGB', (224, 224), color='white')

    def preprocess_data(self, batch, labeled=True):
        default_bert_text = "NaN" # a placeholder text for samples without text
        texts_bert = [sample.get("external_data", default_bert_text) for sample in batch]
        texts_clip = [sample["text"] for sample in batch]
        image_filenames = [sample["image"] for sample in batch]
        if labeled and self.label_binarizer:
            labels = [sample.get("labels", []) for sample in batch]
            default_label = ['None']
            labels = [label if label else default_label for label in labels]
            y = self.label_binarizer.transform(labels)
        else:
            y = None
        
        # BERT processing for 'external_data'
        inputs_bert = self.bert_tokenizer(texts_bert, padding=True, truncation=True, return_tensors="pt")
        outputs_bert = self.bert_model(**inputs_bert)
        bert_embeddings = outputs_bert.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        # CLIP processing for 'text' and 'image'
        images = [self.load_image(os.path.join(self.image_dir, filename)) for filename in image_filenames]
        processed_texts = self.clip_processor(text=texts_clip, return_tensors="pt", padding=True, truncation=True)
        inputs_text = {k: v.to(self.clip_model.device) for k, v in processed_texts.items()}
        text_embeddings = self.clip_model.get_text_features(**inputs_text).cpu().detach().numpy()

        processed_images = self.clip_processor(images=images, return_tensors="pt")
        inputs_image = {k: v.to(self.clip_model.device) for k, v in processed_images.items()}
        image_embeddings = self.clip_model.get_image_features(**inputs_image).cpu().detach().numpy()

        combined_embeddings_clip = np.concatenate((text_embeddings, image_embeddings), axis=1)

        # Combine BERT and CLIP embeddings
        combined_embeddings = np.concatenate((bert_embeddings, combined_embeddings_clip), axis=1)

        if labeled:
            return combined_embeddings, y
        else:
            return combined_embeddings

    def on_epoch_end(self):
        np.random.shuffle(self.indices) 
        
class EarlyFusion:
    def __init__(self, label_tree, data_paths):
        self.data_paths = data_paths
        self.label_binarizer = MultiLabelBinarizer()

        # BERT Initialization
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # CLIP Initialization
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_dir = None  # You can set this when loading data

    def load_and_preprocess_data(self, data_type, sample_size=None):
        # Select the file and image paths based on the data_type
        data_info = self.data_paths.get(data_type)
        if not data_info:
            raise ValueError(f"Invalid data type: {data_type}")

        json_file_path = data_info.get('json_path')
        image_dir = data_info.get('image_dir')

        # Load JSON data
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Check if image files exist
        for sample in data:
            image_path = os.path.join(image_dir, sample.get("image", ""))
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found at {image_path}")

        # Handling sample size
        if sample_size:
            # Ensure that the sample size is not larger than the dataset
            sample_size = min(sample_size, len(data))
            data = np.random.choice(data, sample_size, replace=False).tolist()

        # Handling labels differently based on data_type
        if data_type == 'test':
            # For 'test' data, labels may not be present
            labels = [sample.get("labels", None) for sample in data]
            # Keep only samples with labels (filter out None)
            labels = [label for label in labels if label is not None]
        else:
            # For 'train' and 'dev' data, assign a default label if none exist
            default_label = ['None']
            labels = [sample.get("labels", []) for sample in data]
            labels = [label if label else default_label for label in labels]

        self.label_binarizer.fit(labels)

        return data

    def explore_data(self, sample_size=None, data_type='train', examples_to_show=2):
        data = self.load_and_preprocess_data(data_type, sample_size)

        print(f"Total number of samples: {len(data)}")
        print(f"Total number of unique labels: {len(self.label_binarizer.classes_)}")
        print("Unique labels:", self.label_binarizer.classes_)

        data_generator = DataGenerator(data, batch_size=1, 
                                       bert_model=self.bert_model, 
                                       bert_tokenizer=self.bert_tokenizer,
                                       clip_processor=self.clip_processor, 
                                       clip_model=self.clip_model, 
                                       image_dir=self.image_dir, 
                                       label_binarizer=self.label_binarizer,
                                       data_type=data_type)

        for i in range(examples_to_show):
            sample = [data[i]]  # Wrap the single sample in a list
            print(f"\nSample {i+1} Original Labels:", sample[0].get("labels", []))

            _, labels = data_generator.preprocess_data(sample, labeled=True)
            print("Binarized Labels:", labels)
            combined_embeddings, _ = data_generator.preprocess_data(sample, labeled=True)

            print(f"\nSample {i+1}:")
            print("Text:", sample[0]["text"])
            print("Image:", sample[0]["image"])
            print("Combined Embedding Shape:", combined_embeddings.shape)
            print("Combined Embedding:", combined_embeddings)

    def build_model(self, num_classes, embedding_size=1792, dropout_rate=0.05, learning_rate=0.01):
        # Adjust the input size based on the combined embeddings of BERT and CLIP
        input_layer = Input(shape=(embedding_size,), dtype='float32', name="input")
        dense_layer = Dense(256, activation='relu')(input_layer)
        dropout_layer = Dropout(dropout_rate)(dense_layer)
        output_layer = Dense(num_classes, activation='sigmoid')(dropout_layer)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return self.model

    def train_model(self, save_model_path, batch_size=34, epochs=15, data_type='train', validation_size=0.2, learning_rate=0.01, random_state=42, sample_size=None, is_training=True):
        data = self.load_and_preprocess_data(data_type, sample_size)

        # Split data into training and validation sets
        X_train_indices, X_val_indices = train_test_split(
            range(len(data)), test_size=validation_size, random_state=random_state
        )

        # Generate training and validation data using indices
        train_data = [data[i] for i in X_train_indices]
        val_data = [data[i] for i in X_val_indices]

        # Initialize data generators
        train_generator = DataGenerator(train_data, batch_size, 
                                        self.bert_model, 
                                        self.bert_tokenizer,
                                        self.clip_processor, 
                                        self.clip_model,
                                        data_type,
                                        self.label_binarizer,
                                        is_training=True
                                        )
        
        val_generator = DataGenerator(val_data, batch_size, 
                                        self.bert_model, 
                                        self.bert_tokenizer,
                                        self.clip_processor, 
                                        self.clip_model,
                                        data_type,
                                        self.label_binarizer,
                                        is_training=True
                                        )
        
        for sample_batch in train_generator:
            print("Sample Input Shape:", sample_batch[0].shape, "Type:", sample_batch[0].dtype)
            print("Sample Labels Shape:", sample_batch[1].shape, "Type:", sample_batch[1].dtype)
            break 
    

        # Build the model
        self.build_model(num_classes=len(self.label_binarizer.classes_), dropout_rate=0.05, learning_rate=learning_rate)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

        # Train the model
        history = self.model.fit(
            train_generator, epochs=epochs, validation_data=val_generator, callbacks=[early_stopping], verbose=1
        )

        # Save model
        self.model.save_weights(save_model_path)
        print(f"Model saved at {save_model_path}")

        return history
    
    
    def plot_training_history(self, history):
        plt.figure(figsize=(10, 4))
        
        #plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')

        #plot loss 
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.show()      

    def calculate_hierarchy_distance(self, node1, node2):
        def find_path(tree, node, path=[]):
            if node in tree:
                return path + [node]
            for k, v in tree.items():
                if isinstance(v, dict):
                    new_path = find_path(v, node, path + [k])
                    if new_path:
                        return new_path
            return []

        node1_tuple = (node1,) if isinstance(node1, str) else node1
        node2_tuple = (node2,) if isinstance(node2, str) else node2

        path1 = find_path(self.label_tree, node1_tuple)
        path2 = find_path(self.label_tree, node2_tuple)

        common_length = len(set(path1) & set(path2))
        distance = len(path1) + len(path2) - 2 * common_length
        
        return distance

    def evaluate_model(self, batch_size, save_model_path, num_classes, output_json_path, data_type='dev', learning_rate=0.005):
        # Load and preprocess test data
        dev_data = self.load_and_preprocess_data(data_type)
        test_generator = DataGenerator(dev_data, batch_size, 
                                       bert_model=self.bert_model, 
                                       bert_tokenizer=self.bert_tokenizer,
                                       clip_processor=self.clip_processor, 
                                       clip_model=self.clip_model, 
                                       vision_image_data_path=self.image_dir, 
                                       label_binarizer=self.label_binarizer,
                                       is_training=False)

        # Build the model and load saved weights
        self.build_model(num_classes=len(self.label_binarizer.classes_), dropout_rate=0.5, learning_rate=learning_rate)
        self.model.load_weights(save_model_path)

        # Initialize variables for metrics calculation
        total_precision = 0
        total_recall = 0
        total_samples = 0
        true_labels_all = []
        predicted_labels_all = []

        # Initialize MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=self.label_binarizer.classes_)
        mlb.fit([self.label_binarizer.classes_])

        results = []  # List to store results

        # Iterate over batches in the test generator
        for X, y_true in test_generator:
            y_pred = self.model.predict(X)

            # Iterate over predictions in the batch
            for prediction, true_label in zip(y_pred, y_true):
                gold_labels = [self.label_binarizer.classes_[j] for j in range(len(self.label_binarizer.classes_)) if true_label[j] == 1]
                predicted_labels = [self.label_binarizer.classes_[j] for j in range(len(self.label_binarizer.classes_)) if prediction[j] > 0.5]
                prediction_list = prediction.tolist()

                label_probabilities = {label: float(prob) for label, prob in zip(self.label_binarizer.classes_, prediction_list)}

                true_labels_all.append(gold_labels)
                predicted_labels_all.append(predicted_labels)

                results.append({
                    'true_labels': gold_labels,
                    'predicted_labels': predicted_labels,
                    'predicted_probabilities': label_probabilities
                })

                # Calculate precision and recall
                for predicted_label in predicted_labels:
                    if predicted_label in gold_labels:
                        total_precision += 1
                        total_recall += 1

                total_samples += 1

        # Aggregate metrics over all samples
        average_precision = total_precision / total_samples if total_samples > 0 else 0
        average_recall = total_recall / total_samples if total_samples > 0 else 0
        hierarchical_f1 = 2 * (average_precision * average_recall) / (average_precision + average_recall) if (average_precision + average_recall) != 0 else 0

        true_labels_all_binary = mlb.transform(true_labels_all)
        predicted_labels_all_binary = mlb.transform(predicted_labels_all)
        target_names = self.label_binarizer.classes_

        print("Classification Report:")
        print(classification_report(true_labels_all_binary, predicted_labels_all_binary, target_names=target_names))

        # Save results to JSON file
        result_file_name = f"evaluation_results_{data_type}.json"
        destination_path = os.path.join(output_json_path, result_file_name)
        os.makedirs(output_json_path, exist_ok=True)
        with open(destination_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)

        return hierarchical_f1
  
    def test_model(self, batch_size, save_model_path, output_json_path, data_type='test', learning_rate=0.001):
        test_data = self.load_and_preprocess_data(data_type)
        test_generator = DataGenerator(test_data, batch_size, 
                                       bert_model=self.bert_model, 
                                       bert_tokenizer=self.bert_tokenizer,
                                       clip_processor=self.clip_processor, 
                                       clip_model=self.clip_model, 
                                       vision_image_data_path=self.vision_image_data_path, 
                                       label_binarizer=self.label_binarizer,
                                       is_training=False, labeled=False)

        # Load the trained model weights
        self.build_model(num_classes=len(self.label_binarizer.classes_), dropout_rate=0.5, learning_rate=learning_rate)
        self.model.load_weights(save_model_path)

        predictions = []

        # Iterate over batches in the test generator
        for X, batch_ids in test_generator:
            y_pred = self.model.predict(X)

            # Iterate over predictions in the batch
            for sample_id, prediction in zip(batch_ids, y_pred):
                predicted_labels = [self.label_binarizer.classes_[j] for j in range(len(self.label_binarizer.classes_)) if prediction[j] > 0.001]
                prediction_list = prediction.tolist()

                label_probabilities = {label: float(prob) for label, prob in zip(self.label_binarizer.classes_, prediction_list)}
                
                predictions.append({
                    'id': sample_id,
                    'predicted_labels': predicted_labels,
                    'predicted_probabilities': label_probabilities
                })

        # Save predictions to a JSON file
        result_file_name = "subtask2a_test_pred.json"
        destination_path = os.path.join(output_json_path, result_file_name)
        os.makedirs(output_json_path, exist_ok=True)
        with open(destination_path, 'w') as json_file:
            json.dump(predictions, json_file, indent=4)

        return "Predictions completed."


# hierarchical tree
# the assigned number are hypothetically
label_tree = {
    'Persuasion': {
        'Pathos': {
            'Appeal to Emotion(visual)': 1,
            'Exaggeration/Minimisation': 2,
            'Loaded Language': 3,
            'Flag waving': 4,
            'Appeal to fear/prejudice': 5,
            'Transfer': 6
        },
        'Ethos': {
            'Transfer': 6,
            'Glittering generalities': 7,
            'Appeal to authority': 8,
            'Bandwagon': 9,
            'Ad Hominem': {
                'Name calling/Labelling': 10,
                'Doubt': 11,
                'Smears': 12,
                'Reduction and Hitlerium': 13,
                'Whataboutism': 14
            }
        },
        'Logos': {
            'Repetition': 15,
            'Obfuscation, Intentional vagueness, Confusion': 16,
            'Justification': {
                'Flag waving': 4,
                'Appeal to fear/prejudice': 5,
                'Appeal to Authority': 8,
                'Bandwagon': 9,
                'Slogans': 17
            },
            'Reasoning': {
                'Distraction': {
                    'Whataboutism': 14,
                    'Presenting Irrelevant Data (Red Herring)': 18,
                    'Straw Man': 19
                },
                'Simplification': {
                    'Black-and-white Fallacy/Dictatorship': 20,
                    'Casual Oversimplification': 21,
                    'Thought-terminating cliché': 22
                }
            }
        },
        'None': 23  # a label for empty samples
    }
}

batch_size = 64
num_classes = 23
output_json_path = "path"
save_model_path = "path"

data_paths = {
    'train': {
        'json_path': "path"
        'image_dir': "path"

    },
    'dev': {
        'json_path': 'dev_subtask2a_en.json',
        'image_dir': "path"
    },
    'test': {
        'json_path': 'en_subtask2a_test_unlabeled.json',
        'image_dir': "path"
    }
}

EarlyFusion_classifier2a = EarlyFusion(label_tree, data_paths)

history= EarlyFusion_classifier2a.train_model(save_model_path,
                                          batch_size=64,
                                          epochs=15,
                                          learning_rate=5e-5,
                                          data_type='train'
                                         )

hierarchical_f1 = EarlyFusion_classifier2a.evaluate_model(batch_size,
                                                      save_model_path,
                                                      num_classes,
                                                      output_json_path,
                                                      data_type='dev'
                                                     )
print(f"Average Hierarchical F1: {hierarchical_f1}")

#get predictions on the unlabeled test data
CLIP_meme_classifier2a.test_model(batch_size,
                                save_model_path,
                                output_json_path,
                                data_type='test'
                               )