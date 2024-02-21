import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

class DataGenerator(Sequence):
    def __init__(self, data, batch_size, clip_processor, clip_model, data_type, label_binarizer, max_length=77, is_training=True, is_labeled=True):
        self.data = data
        self.batch_size = batch_size
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.label_binarizer = label_binarizer
        self.max_length = max_length
        self.indices = np.arange(len(self.data))
        self.is_training = is_training  # New attribute to indicate training mode
        self.is_labeled = is_labeled  # New flag to indicate if data is labeled
        self.image_dir = data_paths[data_type]['image_dir']  # Get image directory based on data_type


    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.data[k] for k in batch_indices]
        
        if self.is_labeled:
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
            print(f"Error in loading image: {image_path}. Using a placeholder image.")
            return Image.new('RGB', (224, 224), color='white')

    def preprocess_data(self, batch, labeled=True):
        texts = [sample["text"] for sample in batch]
        image_filenames = [sample["image"] for sample in batch]
        if labeled and self.label_binarizer:
            labels = [sample.get("labels", []) for sample in batch]
            default_label = ['None']
            labels = [label if label else default_label for label in labels]
            y = self.label_binarizer.transform(labels)
        else:
            y = None
        processed_texts = self.clip_processor(text=[text[:self.max_length] for text in texts], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.clip_model.device) for k, v in processed_texts.items()}
        text_embeddings = self.clip_model.get_text_features(**inputs).cpu().detach().numpy()

        images = [self.load_image(os.path.join(self.image_dir, filename)) for filename in image_filenames]
        processed_images = self.clip_processor(images=images, return_tensors="pt")
        image_embeddings = self.clip_model.get_image_features(**processed_images).cpu().detach().numpy()
        combined_embeddings = np.concatenate((text_embeddings, image_embeddings), axis=1)
        
        if labeled:
            return combined_embeddings, y
        else:
            return combined_embeddings

    def on_epoch_end(self):
        np.random.shuffle(self.indices) 
        
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


    
class MultiMemeClassification:
    def __init__(self, label_tree, data_paths):
        self.label_tree = label_tree
        self.label_binarizer = MultiLabelBinarizer()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = None
        self.data_paths = data_paths  #dictionary for data paths by type

    def load_and_preprocess_data(self, data_type, sample_size=None):
        #select the file and image paths based on the data_type
        data_info = self.data_paths.get(data_type)
        if not data_info:
            raise ValueError(f"Invalid data type: {data_type}")

        json_file_path = data_info.get('json_path')
        image_dir = data_info.get('image_dir')

        # load JSON data
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        #check if image files exist
        for sample in data:
            image_path = os.path.join(image_dir, sample.get("image", ""))
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found at {image_path}")

        if sample_size:
            data = np.random.choice(data, sample_size, replace=False)

        #handling labels differently based on data_type
        if data_type == 'test':
            #for 'test' data, labels may not be present
            labels = [sample.get("labels", None) for sample in data]
            #keep only samples with labels (filter out None)
            labels = [label for label in labels if label is not None]
        else:
            #for 'train' and 'dev' data, assign a default label if none exist
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

        data_generator = DataGenerator(data, batch_size=1, clip_processor=self.clip_processor, clip_model=self.clip_model, label_binarizer=self.label_binarizer, data_type=data_type)

        for i in range(examples_to_show):
            sample = [data[i]]  # Wrap the single sample in a list
            combined_embeddings, _ = data_generator.preprocess_data(sample, labeled=True)

            print(f"\nSample {i+1}:")
            print("Text:", sample[0]["text"])
            print("Image:", sample[0]["image"])
            print("Combined Embedding Shape:", combined_embeddings.shape)
            print("Combined Embedding:", combined_embeddings)
            
    def build_model(self, num_classes, embedding_size=1024, dropout_rate=0.5, learning_rate=0.005):
        input_layer = Input(shape=(embedding_size,), dtype='float32', name="input")
        dense_layer = Dense(1024, activation='relu')(input_layer)
        dropout_layer = Dropout(dropout_rate)(dense_layer)
        output_layer = Dense(num_classes, activation='sigmoid')(dense_layer)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return self.model

    
    def train_model(self, save_model_path, batch_size=32, epochs=5, data_type='train', validation_size=0.2, learning_rate=0.01, random_state=42, sample_size=None):
        data = self.load_and_preprocess_data(data_type, sample_size)

        #split data into training and validation sets
        X_train_indices, X_val_indices = train_test_split(
            range(len(data)), test_size=validation_size, random_state=random_state
        )

        #generate training and validation data using indices
        train_data = [data[i] for i in X_train_indices]
        val_data = [data[i] for i in X_val_indices]

        #initialize data generators
        train_generator = DataGenerator(train_data, batch_size, self.clip_processor, self.clip_model, data_type, self.label_binarizer, is_training=True)
        val_generator = DataGenerator(val_data, batch_size, self.clip_processor, self.clip_model, data_type, self.label_binarizer, is_training=True,)

        #build  the model
        self.build_model(num_classes=len(self.label_binarizer.classes_), dropout_rate=0.05, learning_rate=learning_rate)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

        #train the model
        history = self.model.fit(
            train_generator, epochs=epochs, validation_data=val_generator, callbacks=[early_stopping]
        )

        #save model
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
    

    def evaluate_model(self, batch_size, save_model_path, num_classes, output_json_path, data_type='dev',learning_rate=0.005):
        # Load and preprocess test data
        dev_data = self.load_and_preprocess_data(data_type)
        test_generator = DataGenerator(dev_data, batch_size, self.clip_processor, self.clip_model, data_type, self.label_binarizer, is_training=False)

        # Build the model and load saved weights
        self.build_model(num_classes=len(self.label_binarizer.classes_), dropout_rate=0.5, learning_rate=learning_rate)
        self.model.load_weights(save_model_path)

        #initialize variables for metrics calculation
        total_precision = 0
        total_recall = 0
        total_samples = 0
        true_labels_all = []
        predicted_labels_all = []

        #initialize MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=self.label_binarizer.classes_)
        mlb.fit([self.label_binarizer.classes_])

        results = []  # List to store results

        #iterate over batches in the test generator
        for X, y_true, batch_ids in test_generator:
            y_pred = self.model.predict(X)

            #iterate over predictions in the batch
            for sample_id, prediction, true_label in zip(batch_ids, y_pred, y_true):
                gold_labels = [self.label_binarizer.classes_[j] for j in range(len(self.label_binarizer.classes_)) if true_label[j] == 1]
                predicted_labels = [self.label_binarizer.classes_[j] for j in range(len(self.label_binarizer.classes_)) if prediction[j] > 0.5]
                prediction_list = prediction.tolist()

                label_probabilities = {label: float(prob) for label, prob in zip(self.label_binarizer.classes_, prediction_list)}

                true_labels_all.append(gold_labels)
                predicted_labels_all.append(predicted_labels)

                results.append({
                    'id': sample_id,
                    'ture_labels': gold_labels,
                    'predicted_labels': predicted_labels,
                    'predicted_probabilities': label_probabilities  # Convert numpy array to list
                    
                })
                
                #hierarchical evaluation
                for predicted_label in predicted_labels:
                    if predicted_label in gold_labels:
                        total_precision += 1
                        total_recall += 1
                    else:
                        for gold_label in gold_labels:
                            distance = self.calculate_hierarchy_distance(predicted_label, gold_label)
                            if distance is not None and distance > 0:
                                total_precision += 0.5
                                total_recall += 0.5

                total_samples += 1

        #aggregate metrics over all samples
        average_precision = total_precision / total_samples if total_samples > 0 else 0
        average_recall = total_recall / total_samples if total_samples > 0 else 0
        hierarchical_f1 = 2 * (average_precision * average_recall) / (average_precision + average_recall) if (average_precision + average_recall) != 0 else 0

        true_labels_all_binary = mlb.transform(true_labels_all)
        predicted_labels_all_binary = mlb.transform(predicted_labels_all)
        target_names = self.label_binarizer.classes_
        
        print("Classification Report:")
        print(classification_report(true_labels_all_binary, predicted_labels_all_binary, target_names=target_names))
        
        result_file_name = f"subtask2a_dev_pred.json"
        destination_path = os.path.join(output_json_path, result_file_name)
        os.makedirs(output_json_path, exist_ok=True)
        with open(destination_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
            
        return hierarchical_f1


    def test_model(self, batch_size, save_model_path, output_json_path, data_type='test', learning_rate=0.001):
        test_data = self.load_and_preprocess_data(data_type)
        test_generator = DataGenerator(test_data, batch_size, self.clip_processor, self.clip_model, data_type, self.label_binarizer, is_training=False, is_labeled=False)

        # Load the trained model weights
        self.build_model(num_classes=23, dropout_rate=0.5, learning_rate=learning_rate)
        self.model.load_weights(save_model_path)

        predictions = []

        # Iterate over batches in the test generator
        for X, batch_ids in test_generator:
            y_pred = self.model.predict(X)
            print("Raw predictions:", y_pred)
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
                
        result_file_name = "subtask2a_test_pred.json"
        destination_path = os.path.join(output_json_path, result_file_name)
        os.makedirs(output_json_path, exist_ok=True)
        with open(destination_path, 'w') as json_file:
            json.dump(predictions, json_file, indent=4)
            
        return "Predictions completed."


#hierarchical tree
#the assigned number are hypothetically
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
        'None': 23 # a label for empty samples
    }
}

batch_size = 66
num_classes = 23
output_json_path = 'path'
save_model_path = 'path'

data_paths = {
    'train': {
        'json_path': 'path',
        'image_dir': 'path'

    },
    'dev': {
        'json_path': 'path'
        'image_dir': 'path'
    },
    'test': {
        'json_path': 'path',
        'image_dir': 'path'
    }
}

CLIP_meme_classifier2a = MultiMemeClassification(label_tree, data_paths)