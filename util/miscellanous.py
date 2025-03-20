import os
import logging
from pathlib import Path
import inspect

import inspect

import os
import importlib.util
import sys

def find_module(module_name, search_path):
    """
    Cerca un modulo in una directory specificata.
    
    :param module_name: Il nome del modulo da cercare (senza estensione .py).
    :param search_path: Il percorso dove cercare il modulo.
    :return: Il percorso del file del modulo se trovato, altrimenti None.
    """
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if file == f"{module_name}.py":
                return os.path.join(root, file)
    return None

def load_specific_module(search_path, module_name):
    """
    Carica un modulo specifico dal percorso fornito.

    :param search_path: La directory dove cercare il modulo.
    :param module_name: Il nome del modulo da caricare.
    :return: Il modulo caricato o None se non trovato.
    """
    # Trova il percorso del modulo
    module_path = find_module(module_name, search_path)
    
    if not module_path:
        raise ImportError(f"Modulo '{module_name}' non trovato in {search_path}.")
    

    # Costruisci il nome del modulo basato sul percorso
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(module_spec)
    print(module)

    # Carica il modulo
    module_spec.loader.exec_module(module)
    
    return module

