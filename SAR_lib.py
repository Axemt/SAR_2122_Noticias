from contextlib import nullcontext
import json
from pydoc import doc
import string
from unittest import result
from nltk.stem.snowball import SnowballStemmer
import os
import re
import math;
#Jaume te per fer:
#show_stats, reverse_posting, solve_and_show
class SAR_Project:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de noticias
        
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm + ranking de resultado

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [("title", True), ("date", False),
              ("keywords", True), ("article", True),
              ("summary", True)]
    
    
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas 

        """
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list(de les notícies en les quals apareix).
                        # Si se hace la implementacion multifield, se pude hacer un segundo nivel de hashing de tal forma que:
                        # self.index['title'] seria el indice invertido del campo 'title'.
        self.sindex = {} # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {} # hash para el indice permuterm.
        self.docs = {} # diccionario de documentos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = {} # hash de terminos para el pesado, ranking de resultados. puede no utilizarse
        self.news = {} # hash de noticias --> clave entero (newid), valor: la info necesaria para diferenciar la noticia dentro de su fichero (doc_id y posición dentro del documento)
        self.doc_weight_query = {} #diccionario de docID y su peso con la query actual (suma de pesos por termino de query)
        self.tokenizer = re.compile("\W+") # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish') # stemmer en castellano
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()
        #atributs de creació pròpia:
        self.docID = 1 #portem un identificador global del document, inicialment en 1
        self.noticiaID = 1 #portem un identificador global per a cada noticia
        self.frequencies = {} #gastar-ho com a auxiliar per al pesado(weights) que només ho podem calcular una vegada estiguen ja totes les freqüències
    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################

    def set_showall(self, v):
        """

        Cambia el modo de mostrar los resultados.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_snippet(self, v):
        """

        Cambia el modo de mostrar snippet.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v


    def set_stemming(self, v):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v


    def set_ranking(self, v):
        """

        Cambia el modo de ranking por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON RANKING DE NOTICIAS

        si self.use_ranking es True las consultas se mostraran ordenadas, no aplicable a la opcion -C

        """
        self.use_ranking = v

#endregion

#region Indexacion
    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################


    def index_dir(self, root, **args):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Recorre recursivamente el directorio "root" e indexa su contenido, hem de passar-ho sense / inicial, directament és 2015/1 per exemple
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """

        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']
        
        self.index['article'] = {}
        self.weight['article'] = {}
        if self.multifield:
            self.index['title'] = {}
            self.index['date'] = {}
            self.index['keywords'] = {}
            self.index['summary'] = {}   
            #self.weight tindrà un primer nivell d'indexació per al multifield, i després un per a la paraula i dins d'esta per a cada document
            self.weight['title'] = {}
            self.weight['date'] = {}
            self.weight['keywords'] = {}
            self.weight['summary'] = {}  

        for dir, subdirs, files in os.walk(root):
            for filename in files:
                if filename.endswith('.json'):
                    fullname = os.path.join(dir, filename)
                    self.index_file(fullname)
        for field in self.weight.keys():
            for weight_w in self.weight[field].keys():
                idf=math.log10(len(self.news)/(len(self.index[field][weight_w])))
                #if weight_w == 'isla' and field == 'title':
                #    print(idf)
                for weight_doc in self.weight[field][weight_w]:
                    normalized_tf = self.weight[field][weight_w][weight_doc]
                    self.weight[field][weight_w][weight_doc] = normalized_tf*idf
                    #if weight_w == 'isla' and field == 'title':
                    #    print(self.weight[field][weight_w][weight_doc])       
        # Tot l'index generat: fer permuterm
        if self.permuterm:
            self.make_permuterm()
        if self.stemming:
            self.make_stemming()
        #Per fer el càlcul dels pesats, el nombre de noticies en les quals apareix un terme es la longitud de la seua posting list i el nombre d'aparicions en una determinada
        #notícia seria la longitud del segon element de la tupla, perquè té la forma (noticiaID, [pos1, ..., posN])
        #Per fer multifield
        

    def index_file(self, filename):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Indexa el contenido de un fichero.

        Para tokenizar la noticia se debe llamar a "self.tokenize"

        Dependiendo del valor de "self.multifield" y "self.positional" se debe ampliar el indexado.
        En estos casos, se recomienda crear nuevos metodos para hacer mas sencilla la implementacion

        input: "filename" es el nombre de un fichero en formato JSON Arrays (https://www.w3schools.com/js/js_json_arrays.asp).
                Una vez parseado con json.load tendremos una lista de diccionarios, cada diccionario se corresponde a una noticia

        """
        #
        # "jlist" es una lista con tantos elementos como noticias hay en el fichero,
        # cada noticia es un diccionario con los campos:
        #      "title", "date", "keywords", "article", "summary"
        #
        # En la version basica solo se debe indexar el contenido "article"
        # 
        #
        #
        #Enllacem el docID del document en qüestió amb el seu path
        self.docs[self.docID] = filename
        pos = 0 #pos marcarà en quina posició se troba cada notícia en el document del qual forma part, és la posició relativa
        with open(filename) as fh:
            jlist = json.load(fh)
            for noticia in jlist: #és un diccionari
                diccionari = {} #per a cadascuna de les notícies ens creem un diccionari auxiliar que conte les vegades que ha aparegut
                diccionari_posicions = {} #i guardem també les posicions on apareix cada token en eixa notícia
                diccionari['article'] = {}
                diccionari_posicions['article'] = {}
                tokens = {}
                tokens['article'] = self.tokenize(noticia['article']) #tokenitzem la notícia
                self.news[self.noticiaID] = (self.docID, pos, len(tokens['article'])) #guardem una tupla del document on se troba la notícia i la seua posició en ell i la longitud de l'article per fer el ranquing     
                if self.multifield:
                    diccionari['summary'] = {}
                    diccionari['title'] = {}
                    diccionari['keywords'] = {}
                    diccionari_posicions['summary'] = {}
                    diccionari_posicions['title'] = {}
                    diccionari_posicions['keywords'] = {}
                    tokens['summary'] = self.tokenize(noticia['summary'])
                    tokens['keywords'] = self.tokenize(noticia['keywords'])
                    tokens['title'] = self.tokenize(noticia['title'])
                    if noticia['date'] in self.index['date']:
                        self.index['date'][noticia['date']].append(self.noticiaID)
                    else:
                        self.index['date'][noticia['date']] = [self.noticiaID]
                for field in tokens.keys():
                    tokens_field = tokens[field] #ho tenim de manera que és un diccionari amb els tokens per cada camp
                    for index,token in enumerate(tokens_field):
                        diccionari[field][token] = diccionari[field].get(token, 0) + 1
                        if token in diccionari_posicions[field]:
                            diccionari_posicions[field][token].append(index) #si ja existia ho afegim al final
                        else: #si no existeix, creem una llista amb la notícia on l'hem trobat com a primer element
                            diccionari_posicions[field][token] = [index]
                for field in diccionari:
                    for token, aparicions in diccionari[field].items():
                        if token not in self.weight[field]:
                            self.weight[field][token] = {}
                        #el noticiaID no existirà perquè estem considerant-la ara
                        val = aparicions
                        if val != 0:
                            #calculem el nombre d'aparicions
                            val = (math.log10(val) + 1)/ len(tokens[field])

                        self.weight[field][token][self.noticiaID] = val
                        posicions = diccionari_posicions[field][token]
                        if token in self.index[field]:
                            self.index[field][token].append((self.noticiaID, aparicions, posicions))
                        else:
                            self.index[field][token] = [(self.noticiaID, aparicions, posicions)]
                
                pos += 1
                self.noticiaID += 1 #cada vegada ho incrementem perquè no hi haja dues notícies amb el mateix ID
                        
        self.docID += 1 #ho incrementem ja al final 
    
    def tokenize(self, text):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()



    def make_stemming(self):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING.

        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.
        sindex té clave: stem, valor: lista con los terminos que tienen ese stem
        self.stemmer.stem(token) devuelve el stem del token

        """
        self.sindex['article'] = {}
        if self.multifield:
            self.sindex['title'] = {}
            self.sindex['keywords'] = {}
            self.sindex['summary'] = {}  
        
        for token in self.index['article']:
            stem = self.stemmer.stem(token)
            # ocurrStem = 0
            #for (_, aparicions, _) in self.index['article'][token]:
            #    ocurrStem += aparicions
            
            if stem not in self.sindex['article']:
                self.sindex['article'][stem] = [token]
            else:
                self.sindex['article'][stem] = self.sindex['article'][stem] + [token]

            
        if self.multifield:
            fields = ['keywords', 'title', 'summary']
            for f in fields:
                for token in self.index[f]:
                    stem = self.stemmer.stem(token)
                    # ocurrStem = 0
                    # for (_, aparicions, _) in self.index[f][token]:
                    #    ocurrStem += aparicions

                    if stem not in self.sindex[f]:
                        self.sindex[f][stem] = [token]
                    else:
                        self.sindex[f][stem] = self.sindex[f][stem] + [token]
                
        # keyword title summary


    
    def make_permuterm(self):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        """
        
        # self.permuterm layout:
        #
        # 'bc$a' -> 'abc'

        self.ptindex['article'] = {}
        for k in self.index['article'].keys():
        
            # rotate word 
            wordstack = list(k) + ['$']

            exitGuard =  True
            while exitGuard:

                if wordstack[0] == '$':
                    exitGuard = False # exit next iteration

                term = ''.join(wordstack)   
                self.ptindex['article'][term] = self.ptindex['article'].get(term, []) + [k]   
                wordstack.append(wordstack.pop(0))

        
        if self.multifield:

            for field in ['keywords', 'summary', 'title', 'date']:
                self.ptindex[field] = {}

                for k in self.index[field].keys():
                    # rotate word 
                    wordstack = list(k) + ['$']

                    exitGuard =  True
                    while exitGuard:
                    
                        if wordstack[0] == '$':
                            exitGuard = False # exit next iteration

                        term = ''.join(wordstack)   
                        self.ptindex[field][term] = self.ptindex[field].get(term, []) + [k]   
                        wordstack.append(wordstack.pop(0))


    #estadistiques Indexador
    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Muestra estadisticas de los indices
        
        """
        print("========================================")
        print("Nombre de diaris indexats: " + str(len(self.docs)))
        print("----------------------------------------")
        print("Nombre de noticies indexades: " + str(len(self.news)))
        print("----------------------------------------")

        print("TOKENS:")
        if self.multifield:
            for i,j in self.index.items():
                print("nº de tokens en '" + str(i) + "': " + str(len(j)))
        else:
            print("nº de tokens en 'article':" + str(len(self.index['article'].keys())))
        print("----------------------------------------")
        if self.permuterm:
            print("PERMUTERMS:")
            for i,j in self.ptindex.items():
                print("nº de permuterms en '" + str(i) + "':" + str(len(j)))
            print("----------------------------------------")
        if self.stemming:
            print("STEMS:")
            for i,j in self.sindex.items():
                print("nº de stems en '" + str(i) + "':" + str(len(j)))
            print("----------------------------------------")
        if self.positional: # -O
            print("Les consultes posicionals estan permitides")
        else:
            print("Les consultes posicionals NO estan permitides")
        print("========================================")

#endregion
        
#region Recuperacion


    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################
    
    def scan_nestingdepth(self, rest: str) -> str:
        """
        Returns the index of a closing parentesis given a list that is assumed to 
        have a parentesis at position 0 but does not include it
        param: "rest": és una llista amb els tokens des de la posició següent a on s'ha trobat ( fins el final 
        return: torna l'índex on acaba el parèntesi ) que tancaria la query que li hem passat
        """
        
        nest_depth, ct = 1, 0
        #cada ( que trobem suma 1 i quan trobem ) resta un, així quan arribem a 0 és que s'ha tancat el qu s'havia obrt
        while nest_depth != 0:
            if rest[ct] == '(':
                nest_depth += 1
            elif rest[ct] == ')':
                nest_depth -= 1
            ct += 1
        return ct-1

    def solve_query(self, query, prev={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """
        termes = query
        #com per a l'ampliació de parèntesi anem a passar-li una llista com a query en lloc d'un text
        #només en cas que siga text, hem de parsear-lo, puix que ara el mètode s'aprofita per a la crida a solve_query "tradicional"
        #i la recursiva per als parèntesis
        if type(query) == str:
            query = query.replace(")", " ) ").replace("(", " ( ").replace("  ", " ").strip()
            termes = query.split(" ") #separem per espais per tindre tots els termes de la consulta (inclosos AND, NOT i OR)      
        if query is None or len(query) == 0:
            return []
        p1 = []
        i = 1
        if termes[0] == "NOT":
            if self.multifield and ":" in termes[1]:
                [camp, terme] = termes[1].split(":")
                p1 = self.get_posting(terme, camp)
            elif termes[1] == '(':
                # get rest of tokens without first (
                paren_close = self.scan_nestingdepth(termes[2:]) 
                p1 = self.solve_query(termes[2:2+paren_close])
                i += paren_close+1
            else: 
                p1 = self.get_posting(termes[1])
            p1 = self.reverse_posting(p1)
            i += 1
        else:
            if self.multifield and ":" in termes[0]:
                [camp, terme] = termes[0].split(":")
                p1 = self.get_posting(terme, field=camp)
            elif termes[0] == '(':
                # get rest of tokens without first (
                paren_close = self.scan_nestingdepth(termes[1:])
                p1 = self.solve_query(termes[1:1+paren_close])
                i += paren_close+1
            else: 
                p1 = self.get_posting(termes[0])

        while i < len(termes):
            op = ""
            #possible_index_parentesi represetarà l'índex on se trobarà el parèntesi si n'hi ha
            #bé a la dreta d'una AND o OR o bé AND NOT o OR NOT, segons el cas serà i + 1 o i + 2
            possible_index_parentesi = i + 1 
            if i + 1 < len(termes) and termes[i + 1] == "NOT":
                if termes[i] == "AND":
                    op = self.and_not_posting
                else:
                    op = self.or_not_posting
                possible_index_parentesi += 1 #si n'hi ha parèntesi es en i + 2
                nova_i = i + 3
            else:
                if termes[i] == "AND":
                    op = self.and_posting
                else:
                    op = self.or_posting
                nova_i = i + 2 #hem d'indicar a on s'avança, 2 o 3 més segons si tenim NOT o no
            if possible_index_parentesi < len(termes) and termes[possible_index_parentesi] == "(":           
                paren_close = self.scan_nestingdepth(termes[possible_index_parentesi+1:])
                #Fem la crida recursiva des de després del 1r parèntesi fins abans de l'últim parèntesi
                #s'ha de tindre en compte que scan_nestingdepth torna l'índex relatiu, sobre un termes parcial, aleshores hem de sumar-li possbile_index_parentesi
                #per acabar on toca i el mateix per quan calculem nova_i
                p2 = self.solve_query(termes[possible_index_parentesi+1:possible_index_parentesi+1+paren_close])
                nova_i = possible_index_parentesi + paren_close
            elif self.multifield and ":" in termes[nova_i - 1]:
                [camp, terme] = termes[nova_i - 1].split(":")
                p2 = self.get_posting(terme, field=camp)
            else:
                p2 = self.get_posting(termes[nova_i - 1]) #agafem la llista del terme que és un menys de l'element que hem de mirar en la següent iteració
            p1 = op(p1,p2) #en p1 anem guardantnova_i les llistes amb els resultats parcials de la nostra consulta
            #nova_i representa el que haurem de mirar en la següent operació
            i = nova_i
        return p1


    def get_posting(self, term, field='article'):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve la posting list asociada a un termino. 
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list

        """
        #De moment funciona, però quan implementem les ampliacions, com per exemple per a cerques posicionals, que guardem cada noticia i en quina posició, ja no funcionarà,
        #però simplement cal que ho recorrem:
        #posting_list = []
        #for noticia, _ in self.index[term]:
        #   posting_list.append(noticia)
        #return posting_list

        #si no existeix el term en l'índex inveritt tornem la llista buida
        term = term.lower()
        if self.permuterm and '*' in term or '?' in term:
            fullfils_q = self.get_permuterm(term, field=field)
            #Convert list of words into list of postings
            #Separate loops to avoid checking the field condition for every iteration
            left_acum = []
            if field != 'date':
                for t in fullfils_q:
                    left_acum = self.or_posting(left_acum, [ x[0] for x in self.index[field][t] ])
            else:
                for t in fullfils_q:
                    left_acum = self.or_posting(left_acum, self.index[field][t])

            return left_acum

        if self.use_stemming and field != 'date':
            return self.get_stemming(term, field=field) 
        if term not in self.index[field]:
            return []   
        if field == 'date':
            return [x for x in self.index[field][term]]
        else:
            return [x[0] for x in self.index[field][term]] # Posting list normal

    def get_positionals(self, terms, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        pass
        ########################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE POSICIONALES ##
        ########################################################


    def get_stemming(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING

        Devuelve la posting list asociada al stem de un termino.

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list
        AFEGIR CONTEIG DE CADA DOCid PER A CADA STEM
        """
        
        stem = self.stemmer.stem(term)
        
        if self.sindex[field].get(stem, None) != None:
            lstem = self.sindex[field][stem] # llista de paraules amb l'stem
            
            # p1 es la llista composada per l'element 0 de la llista de paraules amb un mateix stem
            p1 = [x[0] for x in self.index[field][lstem[0]]] 
            
            if len(lstem) == 1:
                return p1
            else:
                i = 1
                while i < len(lstem):
                    p1 = self.or_posting(p1, [x[0] for x in self.index[field][lstem[i]]])
                    i += 1
                return p1    
        else: 
            return []
        

    def get_permuterm(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Devuelve la lista de palabras que cumplen una busqueda permuterm.

        param:  "term": termino para recuperar la lista de palabras, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        #(ca*ca) => ca$ca*

        #permuterm => terme orig
        #asaca$c => casaca
        #saca$ca => casaca
        #ca$casa => casaca
        #(casaca)

        # find target permutation for search
        # 'term' contains *, get * to leftmost
        wordstack = list(term) + ['$']

        # permuterm queries with '*' and '?
        while wordstack[0] not in  ['*', '?']:
            wordstack.append(wordstack.pop(0))
        
        # '*/?' in wordstack[0]
        wordstack.append(wordstack.pop(0))
        # term has form ab..$cd..*

        is_qmark = '?' in term
        term = ''.join(wordstack).replace('*','').replace('?','')

        res = []
        left_acum = []
        for k in self.ptindex[field].keys():
            if k.startswith(term):
                if is_qmark and len(k) != len(term) + 1:
                    continue

                left_acum = self.ptindex[field][k] + left_acum

        return left_acum


    def reverse_posting(self, p):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los newid exceptos los contenidos en p

        """
        len_p1 = len(self.news)
        len_p2 = len(p)
        res = []
        p1 = 1 #p1 sempre es igual al nombre al que senyala. Es un comptador
        p2 = 0

        # En usos de pi, ha de ser *menor o igual*, puesto que si es menor estricto perdemos la ultima noticia
        while p1 <= len_p1 and p2 < len_p2:
            if p[p2] > p1:
                res.append(p1)
                p1 +=1
            else:
                p1 +=1
                p2 +=1

        while p1 <= len_p1:
            res.append(p1)
            p1 +=1

        return res


    def and_posting(self, p1, p2): 
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Calcula el AND de dos posting list de forma EFICIENTE
        param:  "p1", "p2": posting lists sobre las que calcular
        return: posting list con los newid incluidos en p1 y p2
        """
        #  p1 = [2,4,8,16,32,64,128]; p2 = [1,2,3,5,8,13,21,34]
        res = []
        idxa = 0
        idxb = 0
        
        while idxa < len(p1) and idxb < len(p2):
            if p1[idxa] == p2[idxb]:
                res.append(p1[idxa])
                idxa += 1
                idxb += 1
            elif p1[idxa] < p2[idxb]:
                idxa += 1
            else:
                idxb += 1
        return res

    def and_not_posting(self, p1, p2): 
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Calcula el ANDNOT de dos posting list de forma EFICIENTE
        param:  "p1", "p2": posting lists sobre las que calcular
        return: posting list con los newid incluidos en p1 y p2
        """
        res = []
        idxa = 0
        idxb = 0
        # if not p1 and not p2: # p1 i p2 no buits VERSIO 1
        while idxa < len(p1) and idxb < len(p2):
            if p1[idxa] == p2[idxb]:
                idxa += 1
                idxb += 1
            elif p1[idxa] < p2[idxb]:
                res.append(p1[idxa])
                idxa += 1
            else:
                idxb += 1
        while idxa < len(p1):
            res.append(p1[idxa])
            idxa += 1

        return res


    def or_posting(self, p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos de p1 o p2

        """
        idxa, idxb = 0,0
        res = []

        while idxa < len(p1) and idxb < len(p2):
            if p1[idxa] < p2[idxb]:
                res.append(p1[idxa])
                idxa += 1
            elif p1[idxa] == p2[idxb]:
                res.append(p1[idxa])
                idxa += 1
                idxb += 1
            else: # p1[idxa] > p2[idxb]
                res.append(p2[idxb])
                idxb += 1
        
        while idxa < len(p1):
            res.append(p1[idxa])
            idxa += 1

        while idxb < len(p2):
            res.append(p2[idxb])
            idxb += 1

        return res


    def or_not_posting(self, p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular
                p2 es la lista sobre la que se aplica NOT


        return: posting list con los newid incluidos de p1 o p2

        """
        return self.or_posting(p1, self.reverse_posting(p2))
#endregion

#region Mostrar resultados


    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################


    def solve_and_count(self, query):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados 

        param:  "query": query que se debe resolver.

        return: el numero de noticias recuperadas, para la opcion -T

        """
        result = self.solve_query(query)
        print("%s\t%d" % (query, len(result)))
        return len(result)  # para verificar los resultados (op: -T)


    def solve_and_show(self, query): #Per als que tenen -Q

        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra informacion de las noticias recuperadas.
        Consideraciones:

        - En funcion del valor de "self.show_snippet" se mostrara una informacion u otra.
        - Si se implementa la opcion de ranking y en funcion del valor de self.use_ranking debera llamar a self.rank_result

        param:  "query": query que se debe resolver.

        return: el numero de noticias recuperadas, para la opcion -T
        
        """
        #region snippet
        def make_snippet(newsID, article):
            #Fase 1
            qList = queryList
            noticia = self.tokenize(article)
            if noticia[0] == 'actualizado':
                noticia = noticia[6:]
                noticia[0] = noticia[0][4:]
            qListAux=[]
            for w in qList:
                subst = w.split(':')
                #article:coche
                if len(subst) == 1 or (len(subst) == 2 and subst[0] == 'article'):
                    if (len(subst) == 2 and subst[0] == 'article'):
                        w = subst[1]
                    if '*' in w or '?' in w:
                        qListAux += self.get_permuterm(w.lower())
                    else:
                        qListAux.append(w)
            qList = qListAux
            #Si la lista de palabras para article está vacia, devolvemos las 25 primeras (busquedas multifield)
            def defaultSnippet():
                snippetRes = ''
                fin = 25
                reached_final = False
                if len(noticia) < fin:
                    fin = len(noticia)
                    reached_final = True
                for w_noticia in noticia[0:fin]:
                    snippetRes += w_noticia + " "
                if reached_final:
                    snippetRes += '...'
                return snippetRes
            
            if len(qList) == 0:
                return defaultSnippet()
            
            #Fase 2
            positionList = []
            wqPosList = []
            for wq in qList:
                if wq in self.index['article'].keys():
                    AuxList = self.index['article'][wq]
                    wqPosList = [x[2] for x in AuxList if x[0] == newsID]
                    if(len(wqPosList) > 0):
                        wqPosList = wqPosList[0]
                    for p in wqPosList:
                        positionList.append(p)
            if len(positionList) == 0:
                return defaultSnippet()
            positionList = sorted(positionList, key= lambda x: x, reverse=False) #Major pos al principi. Menor al final
            #Crear partes de stems...

            #Fase 3
            snippetList = [[positionList[0]]]
            cont = 0
            lastpos = positionList[0]
            if len(positionList) >= 3:
                positionList = positionList[0:4]
            for pi in range(1,len(positionList)):
                if positionList[pi] -lastpos> 30:
                    snippetList.append([])
                    cont+=1
                snippetList[cont].append(positionList[pi])
                lastpos = positionList[pi]

            noticiaWordSnippetList = []
            for l in snippetList:
                rest = 1
                addition = 15
                reached_final = False
                if l[0] == 0:
                    rest = 0
                if self.news[newsID][2] - l[0] <= addition:
                    addition = self.news[newsID][2] - l[0]
                    reached_final = True
                noticiaWordSnippetList += noticia[l[0] - rest:l[-1] + addition]
                if not reached_final:
                    noticiaWordSnippetList += ['...']

            #for wq in qList:
            snippetRes = ""
            for w_noticia in noticiaWordSnippetList:
                snippetRes += w_noticia + " "

            return snippetRes
        #endregion

        print("========================================")
        print("Query: " + query)
        #Llista de les ids de les noticies
        result = self.solve_query(query)
        print("Number of results: " + str(len(result) if result != [] else 0))
        print("----------------------------------------")
        queryAux = query.replace("(", "")
        queryAux = queryAux.replace(")", "")
        command_list = ["NOT", "AND", "OR"]
        queryList = queryAux.split(" ")
        queryList = [x.lower() for x in queryList if x not in command_list]
        self.doc_weight_query = dict()
        if self.use_ranking:
            result = self.rank_result(result, queryList)
        
        if not self.show_all and len(result) >= 10:
            result = result[0:10]
        if self.show_snippet:
            for i in range(0, len(result)):
                docID, newPos, longitud = self.news[result[i]]
                with open(self.docs[docID], "r") as file:
                    jlist = json.load(file)
                    s = "#"+str(i+1) + "\n Score: " + str(self.doc_weight_query.get(result[i],"-")) + "\ndocID: " + str(result[i]) + "\n"
                    if self.multifield:
                        s += "Date: " + jlist[newPos]['date'] + "\n"
                        s += "Title: " + jlist[newPos]['title'] + "\n"
                        s += "Keywords: " + str(jlist[newPos]['keywords'])
                    print(s)
                    print("Snippet: " + make_snippet(result[i],jlist[newPos]['article']))
                    
        else:
            for i in range(0, len(result)):
                docID, newPos, longitud = self.news[result[i]]
                with open(self.docs[docID], "r") as file:
                    jlist = json.load(file)
                    s ="#"+str(i+1)
                    s+="\t(" + str(self.doc_weight_query.get(result[i],"-")) + ")"
                    s+= "\t(" +str(result[i])+")\t" # docID
                    if self.multifield:
                        s += "Date: " + jlist[newPos]['date'] + "\t"
                        s += "Title: " + jlist[newPos]['title'] + "\t"
                        s += "(" + str(jlist[newPos]['keywords']) + ")"
                    print(s)
                    if i < len(result) -1:
                        print("----------------------------------------")
        print("========================================")


    def rank_result(self, result, queryList):
        """
        NECESARIO PARA LA AMPLIACION DE RANKING

        Ordena los resultados de una query.

        param:  "result": lista de resultados sin ordenar
                "query": query, puede ser la query original, la query procesada o una lista de terminos


        return: la lista de resultados ordenada

        """
        queryDict = {'article':[]}
        for w in queryList:
            subst = w.split(':')
            if len(subst) == 2:
                field = subst[0].lower()
                word = subst[1].lower()
                if field != 'date':
                    queryDict[field] = queryDict.get(field, [])
                else:
                    continue
            elif len(subst) == 1:
                field = 'article'
                word = subst[0].lower()

            if field not in self.index:
                print(str(field) + ' field do not exists in our database.')
                return []

            if '*' in word:
                queryDict[field] += self.get_permuterm(word,field)
            else:
                queryDict[field].append(word)
        doc_list = [] #[(docID, weight),(...), ...]
        for doc in result:
            docWeight = 0
            for field in queryDict:
                queryList = queryDict[field]
                for qword in queryList:
                    #print(self.weight[field])
                    if qword in self.weight[field].keys():
                        docWeight += self.weight[field][qword].get(doc,0)
            doc_list.append((doc,docWeight))
            
        self.doc_weight_query = dict(doc_list)
        doc_list = [k for k, v in sorted(doc_list, key=lambda item: item[1], reverse=True)]
        return doc_list
#endregion
