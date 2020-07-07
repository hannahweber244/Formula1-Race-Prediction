class HP_Layer_Optimizer(object):
    
    def __init__ (self, 
                  layer_range = (2,8), #range mindestanzahl layer, max anzahl layer
                  input_start = 52, #anzahl inputneuronen für das Inputlayer
                  output_last = 1, #anzahl outputneuronen für das letzte layer
                  in_out_range = (30,200), #range in der sich die anzahl der Neuronen für die Hiddenlayer bewegen soll
                  random_activation = False,#boolean, die angibt ob activation functions random gepicked werden sollen
                  types = ['linear', 'dropout'],#layertypen, von denen gewählt wird
                  activations = ['relu', 'sigmoid', 'tanh'],#activation functions
                  activation_dist = [0.5,0.25,0.25],#wenn random_activation == False wird hier die Häufigkeit für jede Activation übergeben, nicht implementiert
                  dropout_num = 1,#anzahl von dropoutlayern pro NN
                  cuda = True,#soll mit cuda gearbeitet werden?
                  lr = 0.0001,#lernrate für adam optimizer
                  max_epochs = 5,#anzahl der Trainingsepochen
                  create_combinations = True, #sollen von der Klasse Netze erzeugt werden, oder sollen diese von außen in die Klasse gegeben werden?
                  create_variations = False,#soll von jeder Layeranzahl mehr als ein Netz erzeugt werden?
                  num_variations = 4, #wenn Variationen erzeugt werden sollen wird eine festgelegte Zahl Netze mit 4 Layer, eine festgelegte Anzahl mit 5 Layern usw. erzeugt
                  pure_activations = True #wenn random_activation auf True gesetzt ist kann hier festgelegt werden, dass zusätzlich zu absolut zufällig kombinierten activationfunctions
                  #auch noch Netze erzeugt werden, die nur einheitliche activation functions haben (pure sind)
                  ):
        
        self.__model = None #model, welches trainiert und getestet wird, wird hier zwischengespeichert
        self.__layer_range = layer_range #min und max anzahl von layern in Tupelform
        self.__input_start = input_start#Inputgröße für Startlayer/Inputlayer
        self.__output_last = output_last#Outputgröße für letztes Layer / Outputlayer
        self.__in_out_range = in_out_range#range in der sich die Anzahl der In- und Outputs für die Hiddenlayer bewegen soll (Tupelform)
        self.__layer_types = types#liste welche Layertypen enthält (bspw. linear oder dropout)
        self.__dropout_number = dropout_num#anzahl von layern, die pro Netz ein dropout Layer sein sollen
        self.__random_activation = random_activation#Boolean, ob die activation zufällig gewählt werden soll
        self.__create_variations = create_variations
        self.__number_of_variations = num_variations
        self.__pure = pure_activations
        
        self.__combination_results = {}#dictionary, die dem schlüssel zu einer NN Kombi einen MAE zuordnet
        self.model_specs_combinations = {}#dictionary, welches die jeweiligen NN Kombinationen enthält (einem Schlüssel zugeordnet)
        self.train_data = None#trainingsdaten, die dem Optimizer übergeben werden für die Modelle
        self.test_data = None#developmentdaten, um die Modelle zu testen und Aussagen über die besten Kombinationen zu treffen
        self.lr = lr#lernrate für Modell OPtimizer (default Adam)
        self.max_epochs = max_epochs#Anzahl der Trainingsepochen
        self.cuda = cuda#Boolean, ob mit cuda gearbeitet werden soll oder nicht
        self.opt_combination = {}#dictionary enthält optimale kombination aus möglichen kombinationen
        
        self.__activations = None#entweder Liste (wenn activations random ausgewählt werden sollen), oder dictionary, 
        #wenn activation functions mit einer bestimmten Häufigkeit verwendet werden sollen
        if self.__random_activation:#random pick von activation functions
            self.__activations = activations
        else:#activation function sollen mit einer gewissen häufigkeit ausgewählt werden
            k = {}
            for a in range(len(types)):#zuordnen einer häufihkeit aus activation_dist liste zu jeder activation function aus types
                act = types[a]
                dist = activation_dist[a]
                k[act] = dist
            self.__activations = k
            
        #aufrufen der Funktion, die anhand der übergebenen Parameter Modellkombinationen erzeugt
        if create_combinations:
            print("in constructor bevor die kombinationen erzeugt werden")
            self.__create_combinations()
        else:
            print('NN Kombinationen müssen in Dictionary Form selbst übergeben werden')
        
    def __train(self, epoch, optimizer):
        '''
            funktion übernimmt das Training von dem in self.__model
            zwischengespeicherten NN. 
            Epoch: Jetzige Epoche in der trainiert wird (für coolen print Befehl wichtig)
            Optimizer: Optimizer mit dem Parameter des NN aus self.__model optimiert werden
        '''
        if self.cuda:
            self.__model.cuda()
            self.__model.train()
            batch_id = 0
            for key in self.train_data.keys():
                for data, target in self.train_data[key]:
                    data = data.cuda()
                    target = torch.Tensor(target).unsqueeze(0).cuda()
                    shape = target.size()[1]
                    target = target.resize(shape,1).cuda()
                    optimizer.zero_grad()
                    out = self.__model(data)
                    criterion = nn.MSELoss()
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()
                    ##rint("Train Epoche: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    ##   epoch, batch_id *len(data), len(self.train_data),
                    ##00. * batch_id / len(self.train_data), loss.item()))
                    batch_id +=1
        else:
            self.__model.train()
            batch_id = 0
            for key in self.train_data.keys():
                for data, target in self.train_data[key]:
                    target = torch.Tensor(target).unsqueeze(0)
                    shape = target.size()[1]
                    target = target.resize(shape,1)
                    optimizer.zero_grad()
                    out = self.__model(data)
                    criterion = nn.MSELoss()
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()
                    #print("Train Epoche: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    #    epoch, batch_id *len(data), len(self.train_data),
                    #100. * batch_id / len(self.train_data), loss.item()))
                    batch_id +=1
            
    def __test(self):
        '''
            Funktion, die das Testen des Models aus self.__model auf den übergebenen Dev-Daten 
            (self.test_data) übernimmt und den gesamt MAE für das jeweilige Modell berechnet
        '''
        total = 0
        count = 0
        result_dict = {}
        result = pd.DataFrame(columns = ['target','prediction'])
        help_dict = {}
        if self.cuda:
            self.__model.cuda()
            for key in self.test_data.keys():
                help_dict = {}
                for data, target in self.test_data[key]:
                    self.__model.eval()
                    data = data.cuda()
                    target = torch.Tensor(target).unsqueeze(0)
                    shape = target.size()[1]
                    target = target.resize(shape,1).cuda()
                    out = self.__model(data).cpu()
                    out = out.detach().numpy()
                    target = target.cpu()
                    target = target.detach().numpy()
                    total += abs(out - target[0][0])
                    help_dict[target[0][0]] = out
                    count+=1
                #Auslesen der predicteten Werte A und der zugehörigen targets y
                A = [x[0][0] for x in list((help_dict.values()))]
                y = list(help_dict.keys())
                
                #Anfügen der Werte an Result
                t = pd.DataFrame()
                t['target'] = y
                t['prediction_value'] = A
                t = sqldf.sqldf('''select * from t order by prediction_value ASC''')
                t.reset_index(inplace = True)
                t.rename(columns = {'index':'prediction'}, inplace = True)
                t['prediction'] = t['prediction']+1
                
                
                result = result.append(t, sort = True)
                
        else:
            for raceId in self.test_data.keys():
                help_dict = {}
                for data, target in self.test_data[raceId]:
                    self.__model.eval()
                    target = torch.Tensor(target).unsqueeze(0)
                    shape = target.size()[1]
                    target = target.resize(shape,1)
                    out = self.__model(data)
                    out = out.detach().numpy()
                    target = target.detach().numpy()
                    total += abs(out - target[0][0])
                    help_dict[target[0][0]] = out
                    count+=1
                #Auslesen der predicteten Werte A und der zugehörigen targets y
                A = [x[0][0] for x in list((help_dict.values()))]
                y = list(help_dict.keys())
                
                #Anfügen der Werte an Result
                t = pd.DataFrame()
                t['target'] = y
                t['prediction_value'] = A
                t = sqldf.sqldf('''select * from t order by prediction_value ASC''')
                t.reset_index(inplace = True)
                t.rename(columns = {'index':'prediction'}, inplace = True)
                t['prediction'] = t['prediction']+1
                
                result = result.append(t, sort = True)
        return result
    
    def get_all_information(self):
        
        print('All Model Combinations with encoding:\n', self.model_specs_combinations)
        print('Model Results:\n', self.__combination_results)
        print('Optimale Kombination:\n', self.opt_combination)
        
    def validate_combinations(self):
        
        for key, combination in self.model_specs_combinations.items():
            print("kombination, die jetzt trainiert wird", combination)
            self.__model = NetzDynamic(combination)
            optimizer = optim.Adam(self.__model.parameters(), lr = self.lr)
            #trainieren des modells
            for epoch in range(1,self.max_epochs):
                self.__train(epoch, optimizer)  
            #Aufrufen der Testfunktion
            result = self.__test()
            A = result.prediction.tolist()
            y = result.target.tolist()
            mae = MAE(A,y)
            
            
            
            self.__combination_results[key] = mae
            
        #finden der besten kombination nach minimalstem Error (MAE)
        key_min = min(self.__combination_results.keys(), key=(lambda k: self.__combination_results[k]))
        best_combination = self.model_specs_combinations[key_min]
        best_combination['mae'] = self.__combination_results[key_min]
        self.opt_combination = best_combination
        #self.__combination_overview[key] = specifics
        
        
    def __create_combinations(self):
        '''
            Funktion erzeugt Dictionarys, mit NN Modellspezifikationen, die
            später gegeneinander getestet werden sollen. Kombinationen werden
            in dem Dictionary self.model_specs_combinations unter einem 
            Schlüssel abgespeichert
        '''
        
        if self.__create_variations:
            
            if self.__pure:
                '''
                es werden mehr als ein Netz von einer bestimmten Layeranzahl erzeugt:
                wenn die Layeranzahl Range zwischen 7 und 9 liegt, self.__create_variations = True ist und
                die Variable self.__number_of_variations = 4 ist werden 4 Netze mit 7 Layern und 4 Netze
                mit 8 Layern erzeugt. Diese können dann gegeneinander getestet werden und geben eine bessere
                Übersicht über eine gute Layeranzahl. Wenn self.__pure auf True gesetzt wird, wird aus der 
                Liste mit activationfunctions [relu, sigmoid, tanh] als default 'reine' Netze erzeugt, die immmer
                nur eine der Funktionen enthält (Relu als anfang). Danach werden diese netze kopiert, nur dass die 
                Aktivierungsfunktionen überschrieben werden (ReLu wird durch sigmoid und tanh ersetzt). Zuletzt werden 
                die Funktionen wild gemischt.
                '''
                min_layer = self.__layer_range[0]
                max_layer = self.__layer_range[1]
                if min_layer == max_layer:
                    max_layer += 1
                #pure ist true, deswegen wird zu anfang eine der aktivierungsfunktionen ausgewählt, die fürs ganze netz gewählt wird
                act = random.choice(self.__activations)
                for layer in range(min_layer, max_layer): 
                    variation_counter = 0
                    while variation_counter < self.__number_of_variations:
                        variation_counter +=1 
                        dropout_counter = 0
                        act_count = 0
                        specs_dict = {}
                        middle = layer//2
                        for l in range(layer):
                            layer_specs = []
                            key = act+str(act_count)
                            act_count += 1
                            if dropout_counter == self.__dropout_number:
                                #es wurden schon ausreichend dropout layer erzeugt
                                l_ = random.choice([x for x in self.__layer_types if x not in ['dropout']])
                            else:
                                l_ = random.choice(self.__layer_types)
                                if l_ == 'dropout':
                                    dropout_counter +=1
                                    
                            if l == 0:
                                in_ = self.__input_start
                                if l_ == 'dropout':
                                    dropout_counter = dropout_counter-1
                                    l_ = 'linear'
                                range_start = self.__in_out_range[0]
                                range_end = self.__in_out_range[1]
                                out_ = random.randint(range_start, range_end)
                                layer_specs.append(l_)
                                layer_specs.append(in_)
                                layer_specs.append(out_)
                            else:
                                layer_before = specs_dict[list(specs_dict.keys())[-1]]
                                out_alt = specs_dict[list(specs_dict.keys())[-1]][2]
                                l_type = layer_before[0]
                                i_ltype = -2
                                while l_type == 'dropout':#überprüfen ob vorhergegangenes Layer ein dropout layer war
                                    #sobald das vorhergegangene nicht-dropoutlayer gefunden wurde, wird output größe übernommen
                                    layer_before = specs_dict[list(specs_dict.keys())[i_ltype]]
                                    l_type = layer_before[0]
                                    out_alt = specs_dict[list(specs_dict.keys())[i_ltype]][2]
                                    i_ltype = i_ltype -1
                                if l <= middle: #in der ertsen hälfte nimmt output zu
                                    in_ = out_alt
                                    range_start = out_alt
                                    range_end = self.__in_out_range[1]
                                    out_ = random.randint(range_start, range_end)
                                else:#in zweiter hälfte der layer nimmt output wieder ab
                                    in_ = out_alt
                                    range_start = self.__in_out_range[0]
                                    range_end = out_alt
                                    out_ = random.randint(range_start, range_end)
                                layer_specs.append(l_)
                                layer_specs.append(in_)
                                layer_specs.append(out_)
                            #if l == layer-1:
                            #    l_ = 'linear'
                            #    layer_specs = [l_,in_,self.__output_last]
                                #specs_dict['last'] = layer_specs#layer wird ohne activation gespeichert und als letztes Layer des NN gekennzeichnet
                                
                            if l == 0:
                                specs_dict['first'] = layer_specs
                            else:
                                if specs_dict[list(specs_dict.keys())[-1]][0] == 'dropout':# keine activation function bei layern direkt nach einem dropout layer
                                    key = 'no_activation'+str(act_count)
                                if l == layer -1:
                                    l_ = 'linear'
                                    layer_specs = [l_, in_, self.__output_last]
                                specs_dict[key] = layer_specs
                        key = random.randint(0,10000)
                        while key in list(self.model_specs_combinations.keys()):
                             key = random.randint(0,10000)
                        self.model_specs_combinations[key]= specs_dict   
                        print(specs_dict,'\n')
                        
                used_acts = [act]
                act_count = 0
                netze = list(self.model_specs_combinations.values())
                mixed = []
                if self.__random_activation:
                    p_ = 0
                    for netz in netze:
                        netz_neu = {}
                        for act_alt, layer_spec in netz.items():
                            if act_alt.startswith('no_activation') or act_alt.startswith('first'):
                                netz_neu[act_alt] = layer_spec
                            else:
                                act_neu = random.choice(self.__activations)
                                pp = act_neu+str(p_)
                                p_ += 1
                                netz_neu[pp]= layer_spec
                        mixed.append(netz_neu)
                for activation in self.__activations:
                    if activation in used_acts:
                        continue
                    else:
                        for netz in netze:#alle bisher erzeugten netze werden betrachtet
                            key = random.randint(0,10000)
                            while key in list(self.model_specs_combinations.keys()):
                                key = random.randint(0,10000)
                            netz_neu = {}
                            for act_alt, layer in netz.items():
                                if act_alt.startswith('no_activation') or act_alt.startswith('first'):
                                    netz_neu[act_alt] = layer
                                else:
                                    act_ = activation+str(act_count)
                                    act_count += 1
                                    netz_neu[act_] = layer
                            print(netz_neu,'\n')
                            self.model_specs_combinations[key] = netz_neu
                if len(mixed) != 0:
                    for netz in mixed:
                        print("mixed netz",netz)
                        key = random.randint(0,10000)
                        while key in list(self.model_specs_combinations.keys()):
                            key = random.randint(0,10000)
                        self.model_specs_combinations[key] = netz
                print("alle kombinationen nach create combination:", self.model_specs_combinations)
            else:
                raise Exception('please et pure_activations to True, False not implemented')
                    
        else:
            min_layer = self.__layer_range[0]
            max_layer = self.__layer_range[1]
            if min_layer == max_layer:
                max_layer += 1
            for layer in range(min_layer, max_layer): 
                
                dropout_counter = 0
                act_count = 0
                specs_dict = {}
                middle = layer//2
                for l in range(layer):
                    layer_specs = []
                    if self.__random_activation:#random activation pick ist aktiviert
                        act = random.choice(self.__activations)
                        key = act+str(act_count)
                        act_count += 1
                        if dropout_counter == self.__dropout_number:
                            #es wurden schon ausreichend dropout layer erzeugt
                            l_ = random.choice([x for x in self.__layer_types if x not in ['dropout']])
                        else:
                            l_ = random.choice(self.__layer_types)
                            if l_ == 'dropout':
                                dropout_counter +=1
                                
                        if l == 0:
                            in_ = self.__input_start
                            if l_ == 'dropout':
                                dropout_counter = dropout_counter-1
                                l_ = 'linear'
                            range_start = self.__in_out_range[0]
                            range_end = self.__in_out_range[1]
                            out_ = random.randint(range_start, range_end)
                            layer_specs.append(l_)
                            layer_specs.append(in_)
                            layer_specs.append(out_)
                        else:
                            layer_before = specs_dict[list(specs_dict.keys())[-1]]
                            out_alt = specs_dict[list(specs_dict.keys())[-1]][2]
                            l_type = layer_before[0]
                            i_ltype = -2
                            while l_type == 'dropout':#überprüfen ob vorhergegangenes Layer ein dropout layer war
                                #sobald das vorhergegangene nicht-dropoutlayer gefunden wurde, wird output größe übernommen
                                layer_before = specs_dict[list(specs_dict.keys())[i_ltype]]
                                l_type = layer_before[0]
                                out_alt = specs_dict[list(specs_dict.keys())[i_ltype]][2]
                                i_ltype = i_ltype -1
                            if l <= middle: #in der ertsen hälfte nimmt output zu
                                in_ = out_alt
                                range_start = out_alt
                                range_end = self.__in_out_range[1]
                                out_ = random.randint(range_start, range_end)
                            else:#in zweiter hälfte der layer nimmt output wieder ab
                                in_ = out_alt
                                range_start = self.__in_out_range[0]
                                range_end = out_alt
                                out_ = random.randint(range_start, range_end)
                            layer_specs.append(l_)
                            layer_specs.append(in_)
                            layer_specs.append(out_)
                        #if l == layer-1:
                        #    l_ = 'linear'
                        #    layer_specs = [l_,in_,self.__output_last]
                        #    specs_dict['last'] = layer_specs#layer wird ohne activation gespeichert und als letztes Layer des NN gekennzeichnet
                            
                        #elif l == 0:
                        if l == 0:
                            specs_dict['first'] = layer_specs
                        else:
                            #if l_ == 'dropout':#keine activation function bei dropout layern
                            #    key = 'no_activation'+str(act_count)
                            if specs_dict[list(specs_dict.keys())[-1]][0] == 'dropout':# keine activation function bei layern direkt nach einem dropout layer
                                key = 'no_activation'+str(act_count)
                            if l == layer -1:
                                l_ = 'linear'
                                layer_specs = [l_, in_, self.__output_last]
                            specs_dict[key] = layer_specs
                key = random.randint(0,10000)
                while key in list(self.model_specs_combinations.keys()):
                     key = random.randint(0,10000)
                self.model_specs_combinations[key]= specs_dict   
                print(specs_dict)

                
class HP_Optimizer(object):
    
    def __init__(self, lr_range = (0.0001,0.0001), step_size = 0.0001, max_epochs = (2,2), opt = 'Adam', cuda = True, dynamic = False, dyn_combination = {}):
        
        self.__model = Netz()
        self.__lr = lr_range
        self.__epochs = max_epochs
        self.__optimizer = opt
        self.__steps = step_size
        self.__combination_results = {}
        self.__combination_overview = {}
        self.train_data = None
        self.test_data = None
        self.cuda = cuda
        self.opt_combination = {}
        self.__dynamic = dynamic
        self.__dyn_combination = dyn_combination
        
        
    def validate_combinations(self):
        
        specifics = {}
        if self.__optimizer == 'Adam':
            
            #definieren der range für die lernratenoptimierung
            '''
            
            IST HIER WAS KAPUTT?
            
            '''
            if isinstance(self.__lr, tuple):
                lr_s = self.__lr[0]
                lr_e = self.__lr[1]
                #wurde eine range für die anzahl der epochen übergeben?
                if self.__epochs[0] == self.__epochs[1]:
                    #hyperparamter epochenanzahl wird nicht optimiert
                    max_epoch = self.__epochs[0]
                    if lr_s == lr_e:
                        #es ist keine range für die lernrate gegeben, in der diese optimiert werden soll
                        print('Parameter Epochen und Lernrate können nicht optimiert werden, da kein Intervall übergeben wurde')
                        #setzen des optimizers als Adam und erzeugen des Modells
                        if self.__dynamic:
                            self.__model = NetzDynamic(self.__dyn_combination)
                        else:
                            self.__model = Netz()
                        optimizer = optim.Adam(self.__model.parameters(), lr = lr_s)
                        for epoch in range(1,max_epoch):
                            self.__train(epoch, optimizer)     
                        result = self.__test()
                        A = result.prediction.tolist()
                        y = result.target.tolist()
                        mae = MAE(A,y)
                        specifics = {}
                        specifics['lr'] = lr_s
                        specifics['epochen'] = max_epoch
                        key = random.randint(0,10000)
                        while key in list(self.__combination_results.keys()):
                             key = random.randint(0,10000)
                        self.__combination_results[key] = mae
                        self.__combination_overview[key] = specifics 
                    else:
                        for l in np.arange(lr_s,lr_e, self.__steps):
                            #setzen des optimizers als Adam und erzeugen des Modells
                            if self.__dynamic:
                                self.__model = NetzDynamic(self.__dyn_combination)
                            else:
                                self.__model = Netz()
                            optimizer = optim.Adam(self.__model.parameters(), lr = l)
                            #trainieren des modells
                            for epoch in range(1,max_epoch):
                                self.__train(epoch, optimizer)  
                            result = self.__test()
                            A = result.prediction.tolist()
                            y = result.target.tolist()
                            mae = MAE(A,y)
                            specifics = {}
                            specifics['lr'] = l
                            specifics['epochen'] = max_epoch
                            #abspeichern der gewonnenen informationen (MAE nach lr und anzahl durchgeführter epochen)
                            key = random.randint(0,10000)
                            while key in list(self.__combination_results.keys()):
                                 key = random.randint(0,10000)
                            self.__combination_results[key] = mae
                            self.__combination_overview[key] = specifics
                else:
                    for max_epoch in range(self.__epochs[0], self.__epochs[1]):
                        #definieren der range für die lernratenoptimierung
                        lr_s = self.__lr[0]
                        lr_e = self.__lr[1]
                        #hyperparamter epochenanzahl wird nicht optimiert
                        if lr_s == lr_e:
                            #es ist keine range für die lernrate gegeben, in der diese optimiert werden soll
                            #print('Parameter Epochen und Lernrate können nicht optimiert werden, da kein Intervall übergeben wurde')
                            if self.__dynamic:
                                self.__model = NetzDynamic(self.__dyn_combination)
                            else:
                                self.__model = Netz()
                            optimizer = optim.Adam(self.__model.parameters(), lr = lr_s)
                            for epoch in range(1,max_epoch):
                                self.__train(epoch, optimizer)       
                            result = self.__test()
                            A = result.prediction.tolist()
                            y = result.target.tolist()
                            mae = MAE(A,y)
                            specifics = {}
                            specifics['lr'] = lr_s
                            specifics['epochen'] = max_epoch
                            key = random.randint(0,10000)
                            while key in list(self.__combination_results.keys()):
                                 key = random.randint(0,10000)  
                            self.__combination_results[key] = mae
                            self.__combination_overview[key] = specifics          
                        else:
                            for l in np.arange(lr_s,lr_e, self.__steps):
                                if self.__dynamic:
                                    self.__model = NetzDynamic(self.__dyn_combination)
                                else:
                                    self.__model = Netz()
                                optimizer = optim.Adam(self.__model.parameters(), lr = l)
                                #trainieren des modells
                                for epoch in range(1,max_epoch):
                                    self.__train(epoch, optimizer)  
                                result = self.__test()
                                A = result.prediction.tolist()
                                y = result.target.tolist()
                                mae = MAE(A,y)
                                specifics = {}
                                specifics['lr'] = l
                                specifics['epochen'] = max_epoch
                                #abspeichern der gewonnenen informationen (MAE nach lr und anzahl durchgeführter epochen)
                                key = random.randint(0,10000)
                                while key in list(self.__combination_results.keys()):
                                     key = random.randint(0,10000)
                                self.__combination_results[key] = mae
                                self.__combination_overview[key] = specifics
                #finden der besten kombination nach minimalstem Error (MAE)
                key_min = min(self.__combination_results.keys(), key=(lambda k: self.__combination_results[k]))
                best_combination = self.__combination_overview[key_min]
                best_combination['mae'] = self.__combination_results[key_min]
                self.opt_combination = best_combination
            elif isinstance(self.__lr, list):
                #wurde eine range für die anzahl der epochen übergeben?
                if self.__epochs[0] == self.__epochs[1]:
                    #hyperparamter epochenanzahl wird nicht optimiert
                    max_epoch = self.__epochs[0]
                    if len(self.__lr) ==1:
                        #es ist keine range für die lernrate gegeben, in der diese optimiert werden soll
                        print('Parameter Epochen und Lernrate können nicht optimiert werden, da kein Intervall übergeben wurde')
                        #setzen des optimizers als Adam und erzeugen des Modells
                        if self.__dynamic:
                            self.__model = NetzDynamic(self.__dyn_combination)
                        else:
                            self.__model = Netz()
                        optimizer = optim.Adam(self.__model.parameters(), lr = self.__lr[0])
                        for epoch in range(1,max_epoch):
                            self.__train(epoch, optimizer)     
                        result = self.__test()
                        A = result.prediction.tolist()
                        y = result.target.tolist()
                        mae = MAE(A,y)
                        specifics = {}
                        specifics['lr'] = self.__lr[0]
                        specifics['epochen'] = max_epoch
                        key = random.randint(0,10000)
                        while key in list(self.__combination_results.keys()):
                             key = random.randint(0,10000)
                        self.__combination_results[key] = mae
                        self.__combination_overview[key] = specifics 
                    else:
                        for l in self.__lr:
                            #setzen des optimizers als Adam und erzeugen des Modells
                            if self.__dynamic:
                                self.__model = NetzDynamic(self.__dyn_combination)
                            else:
                                self.__model = Netz()
                            optimizer = optim.Adam(self.__model.parameters(), lr = l)
                            #trainieren des modells
                            for epoch in range(1,max_epoch):
                                self.__train(epoch, optimizer)  
                            result = self.__test()
                            A = result.prediction.tolist()
                            y = result.target.tolist()
                            mae = MAE(A,y)
                            specifics = {}
                            specifics['lr'] = l
                            specifics['epochen'] = max_epoch
                            #abspeichern der gewonnenen informationen (MAE nach lr und anzahl durchgeführter epochen)
                            key = random.randint(0,10000)
                            while key in list(self.__combination_results.keys()):
                                 key = random.randint(0,10000)
                            self.__combination_results[key] = mae
                            self.__combination_overview[key] = specifics
                else:
                    for max_epoch in range(self.__epochs[0], self.__epochs[1]):
                        #definieren der range für die lernratenoptimierung
                        lr_s = self.__lr[0]
                        lr_e = self.__lr[1]
                        #hyperparamter epochenanzahl wird nicht optimiert
                        if len(self.__lr) == 1:
                            #es ist keine range für die lernrate gegeben, in der diese optimiert werden soll
                            #print('Parameter Epochen und Lernrate können nicht optimiert werden, da kein Intervall übergeben wurde')
                            if self.__dynamic:
                                self.__model = NetzDynamic(self.__dyn_combination)
                            else:
                                self.__model = Netz()
                            optimizer = optim.Adam(self.__model.parameters(), lr = self.__lr[0])
                            for epoch in range(1,max_epoch):
                                self.__train(epoch, optimizer)       
                            result = self.__test()
                            A = result.prediction.tolist()
                            y = result.target.tolist()
                            mae = MAE(A,y)
                            specifics = {}
                            specifics['lr'] = self.__lr[0]
                            specifics['epochen'] = max_epoch
                            key = random.randint(0,10000)
                            while key in list(self.__combination_results.keys()):
                                 key = random.randint(0,10000)  
                            self.__combination_results[key] = mae
                            self.__combination_overview[key] = specifics          
                        else:
                            for l in self.__lr:
                                if self.__dynamic:
                                    self.__model = NetzDynamic(self.__dyn_combination)
                                else:
                                    self.__model = Netz()
                                optimizer = optim.Adam(self.__model.parameters(), lr = l)
                                #trainieren des modells
                                for epoch in range(1,max_epoch):
                                    self.__train(epoch, optimizer)  
                                result = self.__test()
                                A = result.prediction.tolist()
                                y = result.target.tolist()
                                mae = MAE(A,y)
                                specifics = {}
                                specifics['lr'] = l
                                specifics['epochen'] = max_epoch
                                #abspeichern der gewonnenen informationen (MAE nach lr und anzahl durchgeführter epochen)
                                key = random.randint(0,10000)
                                while key in list(self.__combination_results.keys()):
                                     key = random.randint(0,10000)
                                self.__combination_results[key] = mae
                                self.__combination_overview[key] = specifics
                #finden der besten kombination nach minimalstem Error (MAE)
                key_min = min(self.__combination_results.keys(), key=(lambda k: self.__combination_results[k]))
                best_combination = self.__combination_overview[key_min]
                best_combination['mae'] = self.__combination_results[key_min]
                self.opt_combination = best_combination
                
        else:
            raise ('No valid optimizer given! Try Adam for example!')
            
    def __train(self, epoch, optimizer):
        if self.cuda:
            self.__model.cuda()
            self.__model.train()
            batch_id = 0
            for key in self.train_data.keys():
                for data, target in self.train_data[key]:
                    data = data.cuda()
                    target = torch.Tensor(target).unsqueeze(0).cuda()
                    shape = target.size()[1]
                    target = target.resize(shape,1).cuda()
                    optimizer.zero_grad()
                    out = self.__model(data)
                    #print("Out: ", out, out.size())
                    #print("Target: ", target, target.size())
                    criterion = nn.MSELoss()
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()
                    #rint("Train Epoche: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    #   epoch, batch_id *len(data), len(self.train_data),
                    #00. * batch_id / len(self.train_data), loss.item()))
                    batch_id +=1
        else:
            self.__model.train()
            batch_id = 0
            for key in self.train_data.keys():
                for data, target in self.train_data[key]:
                    #data = data.cuda()
                    target = torch.Tensor(target).unsqueeze(0)#.cuda()
                    shape = target.size()[1]
                    target = target.resize(shape,1)#.cuda()
                    optimizer.zero_grad()
                    out = self.__model(data)
                    #print("Out: ", out, out.size())
                    #print("Target: ", target, target.size())
                    criterion = nn.MSELoss()
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()
                    #print("Train Epoche: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    #    epoch, batch_id *len(data), len(train_data),
                    #100. * batch_id / len(train_data), loss.item()))
                    batch_id +=1
                    
            
    def __test(self):
        total = 0
        count = 0
        result_dict = {}
        result = pd.DataFrame(columns = ['target','prediction'])
        help_dict = {}
        if self.cuda:
            self.__model.cuda()
            for key in self.test_data.keys():
                #print(key)
                help_dict = {}
                for data, target in self.test_data[key]:
                    self.__model.eval()
                    #files.listdir(path)
                    data = data.cuda()
                    target = torch.Tensor(target).unsqueeze(0)
                    shape = target.size()[1]
                    target = target.resize(shape,1).cuda()
                    out = self.__model(data).cpu()
                    #print(out)
                    out = out.detach().numpy()
                    #out = np.round(out)
                    target = target.cpu()
                    target = target.detach().numpy()
                    #print(data)
                    #print(data["driverId"])
                    total += abs(out - target[0][0])
                    #print("current_position: ", data[0][0].item())
                    #print("Output: ", out)
                    #print("Target: ", target)
                    help_dict[target[0][0]] = out
                    #print("Difference: ", out - target)
                    count+=1
                #Auslesen der predicteten Werte A und der zugehörigen targets y
                A = [x[0][0] for x in list((help_dict.values()))]
                y = list(help_dict.keys())
                
                #Anfügen der Werte an Result
                t = pd.DataFrame()
                t['target'] = y
                t['prediction_value'] = A
                t = sqldf.sqldf('''select * from t order by prediction_value ASC''')
                t.reset_index(inplace = True)
                t.rename(columns = {'index':'prediction'}, inplace = True)
                t['prediction'] = t['prediction']+1
                
                
                result = result.append(t, sort = True)
                
        else:
            for raceId in self.test_data.keys():
                #print(key)
                help_dict = {}
                for data, target in self.test_data[raceId]:
                    self.__model.eval()
                    #files.listdir(path)
                    #data = data.cuda()
                    target = torch.Tensor(target).unsqueeze(0)
                    shape = target.size()[1]
                    target = target.resize(shape,1)#.cuda()
                    out = self.__model(data)#.cpu()
                    #print(out)
                    out = out.detach().numpy()
                    #out = np.round(out)
                    #target = target.cpu()
                    target = target.detach().numpy()
                    #print(data)
                    #print(data["driverId"])
                    total += abs(out - target[0][0])
                    #print("current_position: ", data[0][0].item())
                    #print("Output: ", out)
                    #print("Target: ", target)
                    help_dict[target[0][0]] = out
                    #print("Difference: ", out - target)
                    count+=1
                #Auslesen der predicteten Werte A und der zugehörigen targets y
                A = [x[0][0] for x in list((help_dict.values()))]
                y = list(help_dict.keys())
                
                #Anfügen der Werte an Result
                t = pd.DataFrame()
                t['target'] = y
                t['prediction_value'] = A
                t = sqldf.sqldf('''select * from t order by prediction_value ASC''')
                t.reset_index(inplace = True)
                t.rename(columns = {'index':'prediction'}, inplace = True)
                t['prediction'] = t['prediction']+1
                
                result = result.append(t, sort = True)
        return result
         
            
    def get_all_information(self):
        
        print('Chosen Model:',self.__model)
        print('Learningrate Range:',self.__lr)
        print('Maximum Epochs:', self.__epochs)
        print('Chosen Optimizer:', self.__optimizer)
        print('Result Encoding:', self.__combination_overview)
        print('Results:', self.__combination_results)
        print('Optimale Kombination:', self.opt_combination)
        
    def help(self):
        print('Parameters with defaults:\nlr_range --> (0.0001,0.0001),\nstep_size--> 0.0001,\nmax_epochs-->(2,2),\nopt-->"Adam",\ncuda=True')
        print('lr_range: Tupel with learnrate range')
        print('step_size: float/int for step_size of learnrate')
        print('max_epochs: Tupel with number of epochs range')
        print('opt: Optimizer (by default Adam)')
        print('cuda: True/False if cuda should be used, default = True\n')
        print('Attributes:')
        print('set self.train_data as dictionary with races (form: {raceId: race(dataframe)})')
        print('set self.test_data as dictionary with races (form: {raceId: race(dataframe)})')
        print('self.opt_combination: Dictionary which contains the best combination of the given parameters\n')
        print('Methods:')
        print('call self.validate_combination() to compare all combinations')
        print('get all information/results with self.get_all_information()')