

class analize_dgm():
    def __init__(self):
        pass
    
    def process(self, model, **args):
        self.model, self.d = {}, model
        
        self.temperature_stat(model)


    def temperature_stat(self):
        self.d = {}
        for idx in range(len(self.model.model)):
            self.d[f"temperatue_{idx}"] =  self.model.model[idx].edge_conv.temperature.clone().detach()[0]
        
    
def cat_dicts(a, b):
    return dict(list(a.items()) + list(b.items()))