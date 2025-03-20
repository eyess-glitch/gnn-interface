class EncoderFactory:
    # Mappatura tipo-colonna -> Encoder
    _encoders = {
        "object": None,
        "int64": None,
        "float64": None,
        "category": None,
        "bool": None,             # Tipo booleano (nessun encoder associato)
        "datetime64[ns]": None,   # Tipo datetime (nessun encoder associato)
        "timedelta64[ns]": None,  # Tipo timedelta (nessun encoder associato)
        "complex128": None,       # Tipo complesso (nessun encoder associato)
        "object": None            # Tipo oggetto generico (nessun encoder associato)
    }


    @staticmethod
    def get_encoder(column_type):
    # Restituisci l'istanza corrispondente
        encoder = EncoderFactory._encoders[column_type]
        if encoder is not None:
            return encoder
        return None
   
