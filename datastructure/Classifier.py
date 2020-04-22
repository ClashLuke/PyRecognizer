# -*- coding: utf-8 -*-
"""
Core utils for manage face recognition process
"""
import json
import logging
import os
import pickle
import time
from pprint import pformat

import face_recognition
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datastructure.Person import Person
from utils.util import dump_dataset, load_image_file

log = logging.getLogger()


class AdaptiveLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, dim=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.ones(*self.weight_size))
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None

    @staticmethod
    def _remove_idx(old_tensor, new_size, idx):
        new_size[0] = new_size[0] - 1
        new_tensor = torch.ones(*new_size)
        new_tensor[:idx] = old_tensor[:idx]
        new_tensor[idx:] = old_tensor[idx + 1:]
        return new_tensor

    @staticmethod
    def _add_output(old_tensor, new_size, other):
        new_size[0] = new_size[0] + other
        new_tensor = torch.randn(*new_size)
        new_tensor[:old_tensor.size(0)] = old_tensor[:]
        return new_tensor

    def _change_outputs(self, function, parameter_name, dim, *args):
        old_tensor = getattr(self, parameter_name)
        old_tensor = old_tensor.transpose(0, dim)
        new_tensor = function(old_tensor, list(old_tensor.size()), *args)
        new_tensor = new_tensor.transpose(0, dim)
        setattr(self, parameter_name, new_tensor)

    def change_outputs(self, function, weight_dim, bias_dim, *args):
        self._change_outputs(function, 'weight', weight_dim, *args)
        if self.bias is not None:
            self._change_outputs(function, 'bias', bias_dim, *args)

    def __delitem__(self, idx):
        self.change_outputs(self._remove_idx, 1, 0, idx)

    def __add__(self, other):
        self.change_outputs(self._add_output, 1, 0, other)
        return self

    def forward(self, module_input: torch.Tensor):
        module_input = module_input.transpose(self.dim, -1)
        base_view = [1] * (len(module_input.size()) - 1)
        weight = self.weight.view(*base_view, self.in_features, self.out_features)
        module_input = module_input.unsqueeze(-1)
        output = module_input * weight
        output = output.sum(dim=-2)
        if self.bias is not None:
            bias = self.bias.view(*base_view, self.out_features)
            output = output + bias
        return output


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function_input):
        ctx.save_for_backward(function_input)
        sigmoid = function_input.sigmoid()
        sigmoid.mul_(function_input)
        return sigmoid

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        e_x = x.exp()
        e_x_1 = e_x + 1
        out = e_x_1 + x
        out.mul_(e_x)
        e_x_1.pow_(2)
        out.div_(e_x_1)
        out.mul_(grad_output)
        return out


class ClassificationModel(torch.nn.Module):
    def __init__(self, inputs, *neuron_counts, batch_norm=True, activation=Swish.apply,
                 bias=False, dense_net=False, loss=torch.nn.BCEWithLogitsLoss):
        super().__init__()
        self.dense_net = dense_net
        self.layer_list = [[]]
        input_count = inputs
        for i, neurons in enumerate(neuron_counts[:-1]):
            layer = torch.nn.Linear(input_count, neurons, bias=bias)
            input_count = input_count + neurons if dense_net else neurons

            setattr(self, f'dense_{i}', layer)
            self.layer_list.append([layer])
            if batch_norm:
                setattr(self, f'norm_{i}', torch.nn.BatchNorm1d(neurons))
                self.layer_list[-1].append(layer)
            setattr(self, f'activation_{i}', activation)
            self.layer_list[-1].append(layer)

        self.classifier = AdaptiveLinear(input_count, neuron_counts[-1])
        self.loss = loss
        self.init_weights()

    def init_weights(self):
        def init(module: torch.nn.Module):
            if hasattr(module, "weight"):
                if "norm" not in module.__class__.__name__.lower():
                    torch.nn.init.orthogonal_(module.weight.data)
                else:
                    torch.nn.init.uniform_(module.weight.data, 0.998, 1.002)
            if hasattr(module, "bias"):
                torch.nn.init.constant_(module.bias.data, 0)
        self.apply(init)

    def forward(self, model_input):
        for group in self.layer_list:
            prev_input = model_input
            for layer in group:
                model_input = layer(model_input)
            if self.dense_net:
                model_input = torch.cat([prev_input, model_input], dim=-1)
        output = self.classifier(model_input)
        return output

    def fit(self, inputs, targets=None):
        self.requires_grad_(True)
        mean_loss = 0
        item_count = len(inputs)
        if targets is not None:
            inputs = zip(inputs, targets)
        for src, tgt in inputs:
            out = self.forward(src)
            loss = self.loss(out, tgt)
            loss.backward()
            mean_loss += loss.item() / item_count
        return mean_loss

    def evaluate(self, inputs, targets=None, reduce=torch.argmax):
        self.requires_grad_(False)
        item_count = torch.tensor([len(inputs)]).reshape(1)
        accuracy = torch.zeros(1)
        loss = torch.zeros(1)
        if targets is not None:
            inputs = zip(inputs, targets)
        with torch.no_grad():
            for src, tgt in inputs:
                out = self.forward(src).detach()
                loss += self.loss(out, tgt)
                accuracy += (reduce(out, dim=-1) == targets).sum()
        loss.div_(item_count)
        accuracy.div_(item_count)
        return loss, accuracy


class Classifier:
    """
    Store the knowledge related to the people faces
    """

    def __init__(self, epochs=5, device=None):
        self.training_dir = None
        self.model_path = None
        self.peoples_list = []
        self.classifier = None
        self.parameters = {}
        self.epochs = epochs
        self.device = device if device is not None else (
                torch.device('gpu:0') if torch.cuda.is_available() else torch.device(
                        'cpu'))
        self.batch_size = 16192

    def init_classifier(self):
        """
        Initialize a new classifier after be sure that necessary data are initialized
        """
        if self.classifier is None:
            log.debug("init_classifier | START!")
            if not self.parameters:
                raise ValueError("init_classifier | Mandatory parameter not provided | "
                                 "Init a new KNN Classifier")
            log.debug("init_classifier | Initializing a new classifier ... |" + str(
                    pformat(self.__dict__)))
            self.classifier = ClassificationModel(**self.parameters)

    def load_classifier_from_file(self, timestamp):
        """
        Initialize the classifier from file.
        The classifier file represent the name of the directory related to the
        classifier that we want to load.

        The tree structure of the the model folder will be something like this

         Structure:
        model/
        ├── <20190520_095119>/  --> Timestamp in which the model was created
        │   ├── model.dat       -->  Dataset generated by encoding the faces and
        pickelizing them
        │   ├── model.clf       -->  Classifier delegated to recognize a given face
        │   ├── model.json      -->  Hyperparameters related to the current classifier
        ├── <20190519_210950>/
        │   ├── model.dat
        │   ├── model.clf
        │   ├── model.json
        └── ...

        :param timestamp:
        :return:
        """
        log.debug(
                "load_classifier_from_file | Loading classifier from file ... | File: "
                "{}".format(
                        timestamp))

        # Load a trained KNN model (if one was passed in)
        err = None
        if self.classifier is None:
            if self.model_path is None or not os.path.isdir(self.model_path):
                raise Exception("Model folder not provided!")
            # Adding the conventional name used for the classifier -> 'model.clf'
            filename = os.path.join(self.model_path, timestamp, "model.clf")
            log.debug(
                    "load_classifier_from_file | Loading classifier from file: {"
                    "}".format(
                            filename))
            if os.path.isfile(filename):
                log.debug(
                        "load_classifier_from_file | File {} exist! Loading "
                        "classifier ...".format(
                                filename))
                with open(filename, 'rb') as f:
                    self.classifier = pickle.load(f)
                log.debug("load_classifier_from_file | Classifier loaded!")
            else:
                err = "load_classifier_from_file | FATAL | File {} DOES NOT EXIST " \
                      "...".format(
                        filename)
        if err is not None:
            log.error(
                    "load_classifier_from_file | ERROR: {} | Seems that the model is "
                    "gone "
                    ":/ |"
                    " Loading an empty classifier for training purpose ...".format(err))
            self.classifier = None

    def train(self, X, Y, timestamp):
        """
        Train a new model by the given data [X] related to the given target [Y]
        :param X:
        :param Y:
        :param timestamp:
        """
        log.debug("train | START")
        if self.classifier is None:
            self.init_classifier()

        dump_dataset(X, Y, os.path.join(self.model_path, timestamp))

        start_time = time.time()

        train_input, test_input, train_output, test_output = train_test_split(
                X, Y, test_size=0.25)

        log.debug("train | Training ...")
        loss = -1
        for _ in range(self.epochs):
            loss = self.classifier.fit(train_input, train_output)
        log.debug(f"train | Model Trained. Loss: {loss}")
        log.debug("train | Checking performance ...")
        loss, accuracy = self.classifier.predict(test_input)
        print(f"test | Loss: {loss:.5f} - Accuracy: {accuracy:.5f}")

        return self.dump_model(timestamp=timestamp,
                               classifier=self.classifier), time.time() - start_time

    def tuning(self, X, Y, timestamp):
        """
        Tune the hyperparameter of a new model by the given data [X] related to the
        given target [Y]

        :param X:
        :param Y:
        :param timestamp:
        :return:
        """
        start_time = time.time()
        dump_dataset(X, Y, os.path.join(self.model_path, timestamp))

        train_input, test_input, train_target, test_target = train_test_split(
                X, Y, test_size=0.2)
        train_input, valid_input, train_target, valid_target = train_test_split(train_input, train_target, test_size=0.125)
        # Hyperparameter of the neural network (MLP) to tune
        # Faces are encoded using 128 points
        parameter_space = {
                'neuron_counts': [(128,), (200,), (200, 128,), ],
                }
        epochs = 1
        def train(parameter_names:list, kwargs={}):
            if not parameter_names:
                classifier = ClassificationModel(**kwargs)
                for _ in range(epochs):
                    classifier.fit(train_input, train_target)
                return classifier, classifier.evaluate(test_input, test_target)
            loss = None
            acc = None
            best_param = None
            best_classifier = None
            param = parameter_names.pop(0)
            for value in parameter_space[param]:
                kwargs[param] = value
                classifier, current_loss, current_acc, *params = train(parameter_names,
                                                                       kwargs)
                if loss is None or current_loss < loss or current_acc > acc:
                    loss = current_acc
                    acc = current_acc
                    best_param = params + ((param, best_param),)
                    best_classifier = classifier
            return best_classifier, loss, acc, best_param

        model, loss, acc, params = train(list(parameter_space.keys()))
        params = dict(params)
        log.info("TUNING COMPLETE | DUMPING DATA!")
        log.info(f'Best Parameters: {params}')
        log.info(f'Test-Loss: {loss:.5f} - Test-Accuracy: {acc:.5f}')

        loss, acc = model.evaluate(valid_input, valid_target)
        log.info(f"Valid-Loss: {loss} - Valid-Accuracy: {acc}")
        self.classifier = model

        return self.dump_model(timestamp=timestamp,
                               params=params,
                               classifier=model), time.time() - start_time

    @staticmethod
    def verify_performance(y_test, y_pred):
        """
        Do nothing. Left just in case it's used somewhere.
        """

    def dump_model(self, timestamp, classifier, params=None, path=None):
        """
        Dump the model to the given path, file
        :param params:
        :param timestamp:
        :param classifier:
        :param path:

        """
        log.debug("dump_model | Dumping model ...")
        if path is None:
            if self.model_path is not None:
                if os.path.exists(self.model_path) and os.path.isdir(self.model_path):
                    path = self.model_path
        config = {'classifier_file': os.path.join(timestamp, "model.clf"),
                  'params':          params
                  }
        if not os.path.isdir(path):
            os.makedirs(timestamp)
        classifier_folder = os.path.join(path, timestamp)
        classifier_file = os.path.join(classifier_folder, "model")

        log.debug("dump_model | Dumping model ... | Path: {} | Model folder: {}".format(
                path, timestamp))
        if not os.path.exists(classifier_folder):
            os.makedirs(classifier_folder)

        with open(classifier_file + ".clf", 'wb') as f:
            pickle.dump(classifier, f)
            log.info('dump_model | Model saved to {0}.clf'.format(
                    classifier_file))

        with open(classifier_file + ".json", 'w') as f:
            json.dump(config, f)
            log.info('dump_model | Configuration saved to {0}.json'.format(
                    classifier_file))

        return config

    def init_peoples_list(self, detection_model, jitters, encoding_models,
                          peoples_path=None):
        """
        This method is delegated to iterate among the folder that contains the
        peoples's face in order to
        initialize the array of peoples
        :return:
        """

        log.debug("init_peoples_list | Initializing people ...")
        if peoples_path is not None and os.path.isdir(peoples_path):
            self.training_dir = peoples_path
        else:
            raise Exception("Dataset (peoples faces) path not provided :/")
        # The init process can be parallelized, but BATCH method will perform better
        # pool = ThreadPool(3)
        # self.peoples_list = pool.map(self.init_peoples_list_core, os.listdir(
        # self.training_dir))

        for people_name in tqdm(os.listdir(self.training_dir),
                                total=len(os.listdir(self.training_dir)),
                                desc="Init people list ..."):
            self.peoples_list.append(
                    self.init_peoples_list_core(detection_model, jitters,
                                                encoding_models, people_name))

        self.peoples_list = list(
                filter(None.__ne__, self.peoples_list))  # Remove None

    def init_peoples_list_core(self, detection_model, jitters, encoding_models,
                               people_name):
        """
        Delegated core method for parallelize operation
        :detection_model
        :jitters
        :param people_name:
        :return:
        """
        if os.path.isdir(os.path.join(self.training_dir, people_name)):
            log.debug("Initializing people {0}".format(
                    os.path.join(self.training_dir, people_name)))
            person = Person()
            person.name = people_name
            person.path = os.path.join(self.training_dir, people_name)
            person.init_dataset(detection_model, jitters, encoding_models)
            return person

        log.debug("People {0} invalid folder!".format(
                os.path.join(self.training_dir, people_name)))
        return None

    def init_dataset(self):
        """
        Initialize a new dataset joining all the data related to the peoples list
        :return:
        """
        DATASET = {
                # Image data (numpy array)
                "X": [],
                # Person name
                "Y": []
                }

        for people in self.peoples_list:
            log.debug(people.name)
            for item in people.dataset["X"]:
                DATASET["X"].append(item)
            for item in people.dataset["Y"]:
                DATASET["Y"].append(item)
        return DATASET

    # The method is delegated to try to retrieve the face from the given image.
    # In case of cuda_malloc error (out of memory), the image will be resized
    @staticmethod
    def extract_face_from_image(X_img_path, detection_model, jitters, encoding_models):
        # Load image data in a numpy array
        try:
            log.debug("extract_face_from_image | Loading image {}".format(X_img_path))
            X_img, ratio = load_image_file(X_img_path)
        except OSError:
            log.error("extract_face_from_image | What have you uploaded ???")
            return -2, -2, -1
        log.debug("extract_face_from_image | Extracting faces locations ...")
        try:
            # TODO: Reduce size of the image at every iteration
            X_face_locations = face_recognition.face_locations(
                    X_img, model=detection_model)  # model="cnn")
        except RuntimeError:
            log.error(
                    "extract_face_from_image | GPU does not have enough memory: FIXME "
                    "unload data and retry")
            return None, None, ratio

        log.debug(
                "extract_face_from_image | Found {} face(s) for the given image".format(
                        len(X_face_locations)))

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            log.warning("extract_face_from_image | Seems that no faces was found :( ")
            return -3, -3, ratio

        # Find encodings for faces in the test image
        log.debug(
                "extract_face_from_image | Encoding faces using [{}] jitters "
                "...".format(
                        jitters))
        # num_jitters increase the distortion check
        faces_encodings = face_recognition.face_encodings(
                X_img, known_face_locations=X_face_locations, num_jitters=jitters,
                model=encoding_models)
        log.debug(
                "extract_face_from_image | Face encoded! | Let's ask to the neural "
                "network ...")
        return faces_encodings, X_face_locations, ratio

    def predict(self, X_img_path: str, detection_model: str, jitters: int,
                encoding_models: str,
                distance_threshold: int = 0.45):
        """
        Recognizes faces in given image using a trained KNN classifier

        :param detection_model: can be 'hog' (CPU) or 'cnn' (GPU)
        :param jitters: augmentation data (jitters=20 -> 20x time)
        :param X_img_path: path of the image to be recognized
        :param distance_threshold: (optional) distance threshold for face
        classification. the larger it is,
        the more chance of mis-classifying an unknown person as a known one.
        :return: a list of names and face locations for the recognized faces in the
        image: [(name, bounding box), ...].
                                        For faces of unrecognized persons, the name
                                        'unknown' will be returned.
        """

        if self.classifier is None:
            log.error(
                    "predict | Be sure that you have loaded/trained the neural "
                    "network model")
            return None

        faces_encodings, X_face_locations = None, None
        # Resize image if necessary for avoid cuda-malloc error (important for low gpu)
        # In case of error, will be returned back an integer.
        # FIXME: manage gpu memory unload in case of None
        ratio = 2
        while faces_encodings is None or X_face_locations is None:
            faces_encodings, X_face_locations, ratio = \
                Classifier.extract_face_from_image(
                        X_img_path, detection_model, jitters, encoding_models)
            # In this case return back the error to the caller
            if isinstance(faces_encodings, int):
                return faces_encodings

        # Use the MLP model to find the best matches for the face(s)
        log.debug("predict | Understanding peoples recognized from NN ...")
        closest_distances = self.classifier.predict(faces_encodings)
        log.debug("predict | Persons recognized: [{}]".format(
                closest_distances))

        log.debug("predict | Asking to the neural network for probability ...")
        predictions = self.classifier.predict_proba(faces_encodings)
        pred = []
        for prediction in predictions:
            pred.append(
                    dict([v for v in sorted(zip(self.classifier.classes_, prediction),
                                            key=lambda c: c[1], reverse=True)[
                                     :len(closest_distances)]]))
        log.debug("predict | Predict proba -> {}".format(pred))
        face_prediction = []
        for i in range(len(pred)):
            element = list(pred[i].items())[0]
            log.debug("pred in cycle: {}".format(element))
            face_prediction.append(element)
            # log.debug("predict | *****MIN****| {}".format(min(closest_distances[0][
            # i])))
        log.debug("Scores -> {}".format(face_prediction))

        _predictions = []
        scores = []
        if len(face_prediction) > 0:
            for person_score, loc in zip(face_prediction, X_face_locations):
                if person_score[1] < distance_threshold:
                    log.warning(
                            "predict | Person {} does not outbounds threshold {}<{"
                            "}".format(
                                    pred, person_score[1], distance_threshold))
                else:
                    log.debug("predict | Pred: {} | Loc: {} | Score: {}".format(
                            person_score[0], loc, person_score[1]))
                    if ratio > 0:
                        log.debug(
                                "predict | Fixing face location using ratio: {}".format(
                                        ratio))

                        x1, y1, x2, y2 = loc
                        # 1200 < size < 1600
                        if ratio < 1:
                            ratio = pow(ratio, -1)
                        x1 *= ratio
                        x2 *= ratio
                        y1 *= ratio
                        y2 *= ratio
                        loc = x1, y1, x2, y2

                    _predictions.append((person_score[0], loc))
                    scores.append(person_score[1])
            log.debug("predict | Prediction: {}".format(_predictions))
            log.debug("predict | Score: {}".format(scores))

        if len(_predictions) == 0 or len(face_prediction) == 0:
            log.debug("predict | Face not recognized :/")
            return -1

        return {"predictions": _predictions, "scores": scores}
