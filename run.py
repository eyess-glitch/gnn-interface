import os
import sys
import json
import logging
import torch
import importlib
from pathlib import Path
from loguru import logger
from util.miscellanous import *
from builder.homogenous_graph_builder import HomogenousGraphBuilder
from torch_geometric.nn import to_hetero

from data_loader.data_loader_registry import DataLoaderRegistry
from model.model_registry import ModelRegistry
from training.training_registry import TrainingRegistry
from evaluator.evaluator_registry import EvaluatorRegistry

def main(config):
    # Configura il logging
    logger.add(sys.stderr, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", colorize=True)
    
    logging.info("Analisi dei dati...")
    data_config = config["data"]
    graph_type = data_config["graph_type"]
    data_folder_path = data_config["folder_path"]
    
    builder = None

    # Costruisci il grafo
    # eventualmente si può inferire il tipo ? 
    if graph_type == "homogenous":
        print("here")
        builder = HomogenousGraphBuilder(data_folder_path)
    # da aggiustare
    else:
        folder_name = Path(data_folder_path).name
        builder_class_name = find_class_file("./builder", folder_name)

        if not builder_class_name:
            logger.error(f"Nessun builder trovato per {folder_name}")
            return
        
        builder_module = importlib.import_module(f"builder.{builder_class_name}")
        builder_class = getattr(builder_module, builder_class_name)
        builder = load_component(data_config, builder_class)

    graph_data = builder.build_graph()
    task_type = config["training"]["task"]

    if task_type == "classification":
        # Assuming data.x is a tensor and the last column is the label column
        # Extract the last column (assumed to be the labels)
        graph_data.y = graph_data.x[:, -1]  # The last column becomes y
        # Remove the last column from data.x
        graph_data.x = graph_data.x[:, :-1] 

    loader_name = data_config['data_loader']
    search_path = './data_loader'  # Percorso in cui cercare i moduli (deve essere specificato correttamente)

    try:
        # Carica il modulo dinamicamente
        loaded_module = load_specific_module(search_path, loader_name + '_data_loader')
        logger.info(f"Modulo '{loader_name}_data_loader' caricato con successo.")
    except ImportError as e:
        logger.error(str(e))
        return
    
    data_loader = DataLoaderRegistry.get_data_loader(**data_config)
    data_loader.set_data(graph_data)
    train_data_loader = data_loader.get_loader("train")
    
    # Carica il modello
    logging.info("Caricamento del modello...")
    model_config = config["model"]
    model_name = model_config["model_name"]
    model_search_path = "./model"  # Percorso per i moduli del modello

    try:
        # Carica il modulo del modello dinamicamente
        loaded_model_module = load_specific_module(model_search_path, model_name)
        logger.info(f"Modulo '{model_name}_model' caricato con successo.")
    except ImportError as e:
        logger.error(f"Errore nel caricamento del modulo del modello: {str(e)}")
        return

    # Ottieni il modello dal registro
    model = ModelRegistry.get_model(model_name, **model_config)

    if graph_type.lower() != "homogenous":
        # gestire aggregazione custom
        model = to_hetero(model, graph_data.metadata())

    logging.info(f"Modello caricato: {model}")
    logging.info("Caricamento della strategia di addestramento...")
    
    training_config = config["training"]
    trainer_name = model_name + "_trainer"
    model_search_path = "./training"  # Percorso per i moduli del modello

    try:
        # Carica il modulo del modello dinamicamente
        loaded_trainer_module = load_specific_module(model_search_path, trainer_name)
        logger.info(f"Modulo '{loaded_trainer_module}_model' caricato con successo.")
    except ImportError as e:
        logger.error(f"Errore nel caricamento del modulo del modello: {str(e)}")
        return

    trainer = TrainingRegistry.get_trainer(trainer_name, **training_config)
    trainer.set_param("data_loader", train_data_loader)
    logging.info(f"Strategia di addestramento caricata: {trainer}")

    trainer.train(model)
    
    # Verifica se "save_path" è presente nel dizionario data_config
    if "save_path" not in data_config:
        save_dir = "model_weights"  # Directory predefinita
        save_path = os.path.join(save_dir, "{model_name}_model.pth")
    else:
        save_dir = os.path.dirname(data_config["save_path"])  # Estrai la directory da save_path
        save_path = data_config["save_path"]

    # Crea la directory se non esiste
    os.makedirs(save_dir, exist_ok=True)

    # Salva il modello
    torch.save(model.state_dict(), save_path)


    """
    if "evaluation" in config:
        if "metric" in config["evaluation"]:
            metric = config["evaluation"]["metric"]

    try:
        # Carica il modulo dinamicamente
        loaded_module = load_specific_module("./evaluator", metric + '_evaluator')
        logger.info(f"Modulo '{loader_name}_data_loader' caricato con successo.")
    except ImportError as e:
        logger.error(str(e))
        return


    evaluator = EvaluatorRegistry.get_evaluator("metric")
    test_data_loader = data_loader.get_loader("test")
    evaluator.eval(model, test_data_loader)
    """

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Esegui il training di un modello specificato nel file JSON.")
    parser.add_argument("--config", type=str, required=True, help="Percorso al file di configurazione JSON.")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = json.load(f)
        main(config)
    except FileNotFoundError:
        logger.error(f"File di configurazione non trovato: {args.config}")
    except json.JSONDecodeError as e:
        logger.error(f"Errore nel parsing del file JSON: {e.msg} (linea {e.lineno}, colonna {e.colno})")
