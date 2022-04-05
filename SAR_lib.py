from contextlib import nullcontext
import json
import string
from unittest import result
from nltk.stem.snowball import SnowballStemmer
import os
import re
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
        if self.multifield:
            self.index['title'] = {}
            self.index['date'] = {}
            self.index['keywords'] = {}
            self.index['summary'] = {}   

        for dir, subdirs, files in os.walk(root):
            for filename in files:
                if filename.endswith('.json'):
                    fullname = os.path.join(dir, filename)
                    self.index_file(fullname)

        #Per fer el càlcul dels pesats, el nombre de noticies en les quals apareix un terme es la longitud de la seua posting list i el nombre d'aparicions en una determinada
        #notícia seria la longitud del segon element de la tupla, perquè té la forma (noticiaID, [pos1, ..., posN])
        #Per fer multifield

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################
        

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
        pos = 1 #pos marcarà en quina posició se troba cada notícia en el document del qual forma part
        with open(filename) as fh:
            jlist = json.load(fh)
            for noticia in jlist: #és un diccionari
                diccionari = {} #per a cadascuna de les notícies ens creem un diccionari auxiliar que conte les vegades que ha aparegut
                diccionari_posicions = {} #i guardem també les posicions on apareix cada token en eixa notícia
                self.news[self.noticiaID] = (self.docID, pos) #guardem una tupla del document on se troba la notícia i la seua posició en ell
                tokens = self.tokenize(noticia['article']) #tokenitzem la notícia
                for index, token in enumerate(tokens):
                    if token in diccionari_posicions:
                        diccionari_posicions[token].append(index) #si ja existia ho afegim al final
                        diccionari[token] = diccionari[token] + 1
                        #Per a cerques posicionals:
                        #aux = self.index[token]
                        #Ara faltaria saber com mirar si la notícia ja està dins o no, perquè lo que tenim és una llista de tuples, hauríem de recórrer-la tota? 
                        #S'hauria de discutir, preguntar-li en classe
                    else: #si no existeix, creem una llista amb la notícia on l'hem trobat com a primer element
                        diccionari_posicions[token] = [index]
                        diccionari[token] = 1
                        #Per a cerques posicionals: Tal volta és millor idea utilitzar un diccionari per a cada terme i té com a clau noticiaID i com a valor la llista de posicions
                        # self.index[token] = [(self.noticiaID, [idParaula])]       
                for token, aparicions in diccionari.items():
                    posicions = diccionari_posicions[token]
                    if token in self.index['article']:
                        self.index['article'][token].append((self.noticiaID, aparicions, posicions))
                    else:
                        self.index['article'][token] = [(self.noticiaID, aparicions, posicions)]
                pos += 1
                self.noticiaID += 1 #cada vegada ho incrementem perquè no hi haja dues notícies amb el mateix ID
        self.docID += 1 #ho incrementem ja al final
        pos = 1 #cada vegada pose la posició a 1 perquè siga la posició relativa de la notícia dins el document    
    
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

        self.stemmer.stem(token) devuelve el stem del token

        """
        
        pass
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################


    
    def make_permuterm(self):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        """
        pass
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################



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
            for i,j in self.index:
                print("nº de tokens en '" + str(i) + "':" + str(len(j)))
        else:
            print("nº de tokens en 'article':" + str(len(self.index['article'].keys())))
        print("----------------------------------------")
        if self.permuterm:
            print("PERMUTERMS:")
            for i,j in self.ptindex:
                print("nº de permuterms en '" + str(i) + "':" + str(len(j)))
            print("----------------------------------------")
        if self.stemming:
            print("STEMS:")
            for i,j in self.sindex:
                print("nº de permuterms en '" + str(i) + "':" + str(len(j)))
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


    def solve_query(self, query, prev={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """

        if query is None or len(query) == 0:
            return []
        termes = query.split(" ") #separem per espais per tindre tots els termes de la consulta (inclosos AND, NOT i OR)
        p1 = []
        i = 1
        if termes[0] == "NOT":
            p1 = self.index[termes[1]]
            p1 = self.reverse_posting(p1)
            i += 1
        else:
            p1 = self.index[termes[0]]
        while i < len(termes):
            op = ""
            if termes[i + 1] == "NOT":
                if termes[i] == "AND":
                    op = self.and_not_posting
                else:
                    op = self.or_not_posting
                nova_i = i + 3
            else:
                if termes[i] == "AND":
                    op = self.and_posting
                else:
                    op = self.or_posting
                nova_i = i + 2 #hem d'indicar a on s'avança, 2 o 3 més segons si tenim NOT o no
            p2 = self.index[termes[nova_i - 1]] #agafem la llista del terme que és un menys de l'element que hem de mirar en la següent iteració
            p1 = op(p1,p2) #en p1 anem guardant les llistes amb els resultats parcials de la nostra consulta
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
        docs = []
        for x in self.index[field][term]:
            docs.append(x[0])
        return docs #si no existeix el term en l'índex inveritt tornem la llista buida

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

        """
        
        stem = self.stemmer.stem(term)

        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################


    def get_permuterm(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        ##################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA PERMUTERM ##
        ##################################################




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
        p1 = 0 #p1 sempre es igual al nombre al que senyala. Es un comptador
        p2 = 0
        while p1 < len_p1 and p2 < len_p1:
            if result[p2] > p1:
                res.append(p1)
                p1 +=1
            else:
                p1 +=1
                p2 +=1
        while p1 < len_p1:
            res.append(p1)
            p1 +=1

        return res


    def and_posting(self, p1, p2): #VIOLETA
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
            print(idxa)
            if p1[idxa] == p2[idxb]:
                res.append(p1[idxa])
                idxa += 1
                idxb += 1
            elif p1[idxa] < p2[idxb]:
                idxa += 1
            else:
                idxb += 1
        return res

    def and_not_posting(self, p1, p2): #VIOLETA
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

        p2_reverse = self.reverse_posting(p2)

        res = self.or_posting(p1, p2_reverse)
            

        return res


    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se propone por si os es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos de p1 y no en p2

        """

        
        pass
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################
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

        #AVIS!!!!! L'accés al diccionari pot estar MAL i faltar algun [1]
        print("========================================")
        print("Query: " + query)
        #Llista de les ids de les noticies
        result = self.solve_query(query)
        print("Number of results: " + len(result))
        if self.use_ranking:
            result = self.rank_result(result, query)   
        if self.show_snippet:
            for i in range(0, len(result)):
                s = "#"+str(i+1) + "\t (" + str(self.weight[result[i]]) + ")" + " (" + str(result[i]) + ")"
                if self.multifield:
                    if self.index.get("date", None) != None:
                        s += " (" + self.index['date'][result[i]] + ")"
                    if self.index.get("title", None) != None:
                        s += self.index['title'][result[i]]
                    if self.index.get("keywords", None) != None:
                        s += str(self.index['keywords'][result[i]][1])  #El [1] es per a agafar la llista potser estiga mal
                print(s)
        else:
            for i in range(0, len(result)):
                print("#"+str(i+1))
                print("Score: " + str(self.weight[result[i]])) 
                print(result[i])
                if self.multifield:
                    if self.index.get("date", None) != None:
                        print(self.index['date'][result[i]])
                    if self.index.get("title", None) != None:
                        print(self.index['title'][result[i]])
                    if self.index.get("keywords", None) != None:
                        print(str(self.index['keywords'][result[i]][1])) #El [1] es per a agafar la llista potser estiga mal
                print(s)
                if i < len(result) -1:
                    print("----------------------------------------")
        print("========================================")

    def rank_result(self, result, query):
        """
        NECESARIO PARA LA AMPLIACION DE RANKING

        Ordena los resultados de una query.

        param:  "result": lista de resultados sin ordenar
                "query": query, puede ser la query original, la query procesada o una lista de terminos


        return: la lista de resultados ordenada

        """

        pass
        
        ###################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE RANKING ##
        ###################################################
#endregion
