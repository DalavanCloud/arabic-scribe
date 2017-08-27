# -*- coding: utf-8 -*-
import arabic_reshaper
import codecs
import numpy as np
import math
import random
import os
import cPickle as pickle
import xml.etree.ElementTree as ET
import re


from utils import *

class DataLoader():
    def __init__(self, args, logger, limit = 500):
        self.args = args
        self.preprocessing_type = args.preprocessing_type
        self.visual_dir = args.visual_dir
        self.datasetAnalysis = args.datasetAnalysis
        self.data_dir = args.data_dir
        self.alphabet = args.alphabet
        self.unknowntoken = args.unknowntoken
        self.batch_size = args.batch_size
        self.tsteps = args.tsteps
        self.data_scale = args.data_scale # scale data down by this factor
        self.ascii_steps = args.tsteps/args.tsteps_per_ascii
        self.tsteps_per_ascii = args.tsteps_per_ascii
        self.logger = logger
        self.filter = args.filter
        self.limit = limit # removes large noisy gaps in the data
        self.idx_path = os.path.join(self.data_dir, "idx.cpkl");
        self.pointer_path = os.path.join(self.data_dir, "pointer.cpkl")
        data_file = os.path.join(self.data_dir, "strokes_training_data.cpkl")
        stroke_dir = self.data_dir + "/lineStrokes"
        ascii_dir = self.data_dir + "/ascii"

        if not (os.path.exists(data_file)) :
            self.logger.write("\tcreating training data cpkl file from raw source")

            self.preprocess(stroke_dir, ascii_dir, data_file)

        self.load_preprocessed(data_file)
        self.reset_batch_pointer()


    def preprocess(self, stroke_dir, ascii_dir, data_file):

        # def preprocessDataset():
        # function to read each individual xml file
        def getStrokes(filename):
            # Each XML File represents an entire line in the form
            # Uses an XML Parser that creates a tree of the xml file

            tree = ET.parse(filename)
            root = tree.getroot()

            results = []
            x_max = -1e20
            y_max = -1e20
            for child in root:
                pointslist = []
                if (child.tag == "{http://www.w3.org/2003/InkML}trace"):
                    points = child.text.split(",")
                    for point in points:
                        x_offset = -1e20
                        y_offset = -1e20
                        x, y = point.split(" ")
                        x_offset = max(x_offset, float(x))
                        y_offset = max(y_offset, float(y))
                        x_max = max(x_max, float(x))
                        y_max = max(y_max, float(y))
                        pointslist.append([x_offset, y_offset])
                    results.append(pointslist)

            results = extraPreprocessing(results)

            # Createss a padding
            x_max = x_max + 100.0
            y_max = y_max + 100.0
            import random
            ran=random.randint(0, 99)
            if(ran < 2):
                Fname= ((filename.split(".")[1]).split("/")[5]).split(".")[0]
                visualize(args=self.args ,filename=Fname,results=results)

            for i in range(0, len(results)):
                for j in range(0, len(results[i])):
                    results[i][j] = [results[i][j][0] - x_max, results[i][j][1] - y_max]


            # result is basically a 2D array each outer array represents a Stroke and each inner array represents a stroke
            # result[0][0] represents the first point in the first stroke
            # after that passes the result to convert_stokes_to_array

            return results

        def extraPreprocessing(results):

            if(self.preprocessing_type == "dotsRepositioned"):
                listOfStrokes = []
                maxm = []
                minm = []
                for stroke in results:
                    temp = [map(int, list(zip(*stroke)[0])), map(int, list(zip(*stroke)[1]))]
                    maxm.append(max(temp[0]))
                    minm.append(min(temp[0]))
                    listOfStrokes.append(temp)
                ourIndices = []
                margin = 5

                for i in range(len(listOfStrokes)):
                    maxlist = [j for j, x in enumerate(maxm[:i]) if x + margin > maxm[i]]
                    minlist = [j for j, x in enumerate(minm[:i]) if x - margin < minm[i]]
                    commonElement = list(set(maxlist).intersection(minlist))
                    commonElement.sort()
                    if (maxlist and minlist and commonElement):
                        ourIndices.append([i, commonElement[-1]])

                filtered = []
                NewlistOfStrokes = []
                for i in range(len(listOfStrokes)):
                    if(filtered.__contains__(i)):
                        continue
                    stroke = listOfStrokes[i]
                    if (not any(e[1] == i for e in ourIndices)):
                        NewlistOfStrokes.append(stroke)
                        continue
                    for index in ourIndices:
                        if (index[1] == i):
                            for j, x in enumerate(stroke[0]):
                                if (x < maxm[index[0]] + margin and x > maxm[index[0]] - margin):
                                    filtered.append(index[0])
                                    NewlistOfStrokes.append([stroke[0][:j + 2], stroke[1][:j + 2]])
                                    NewlistOfStrokes.append(listOfStrokes[index[0]])
                                    stroke = [stroke[0][j:], stroke[1][j:]]
                                    break
                    NewlistOfStrokes.append(stroke)
                results = []
                for stroke in NewlistOfStrokes:
                    pointslist = []
                    if (len(stroke[0]) == 1): continue
                    for i in range(len(stroke[0])):
                        point = [stroke[0][i], stroke[1][i]]
                        pointslist.append(point)
                    results.append(pointslist)
            return results



        # function to read each individual xml file
        def getUnicode(filename,unknowntoken):
            # Gets the Word as Unicode
            indexs = []
            tree = ET.parse(filename)
            root = tree.getroot()
            unshapedUnicode = root[2][0][0][0].get('value')
            shapedUnicode = arabic_reshaper.reshape(unshapedUnicode)
            if(len(shapedUnicode)!= len(unshapedUnicode)):
                chars = list(unshapedUnicode)
                for index, char in enumerate(chars):
                    if (not unknowntoken.__contains__(char)):
                        indexs.append(index)
            else:
                return shapedUnicode

            for index in indexs:
                shapedUnicode = shapedUnicode[:index] + unshapedUnicode[index] + shapedUnicode[index:]
            return shapedUnicode


        # converts a list of arrays into a 2d numpy int16 array
        def convert_stroke_to_array(stroke,unicode):
            n_point = 0
            for i in range(len(stroke)):
                n_point += len(stroke[i])

            # # Creates a  matrix of n columns and 3 rows
            stroke_data = []

            prev_x = 0
            prev_y = 0

            pointInterpolationThreshold = self.tsteps_per_ascii
            avgPointPerCharacter =  n_point/len(unicode)

            if (avgPointPerCharacter > pointInterpolationThreshold):    # No Interpolation

                for i in range(len(stroke)):
                    for j in range(len(stroke[i])):

                        # Creates each point relative to the one before it
                        # The first index [counter,0] represents the counterth x relative to previous x
                        x = int(stroke[i][j][0]) - prev_x

                        # The second index [counter,1] represents the counterth y relative to the previous y
                        y = int(stroke[i][j][1]) - prev_y


                        prev_x = int(stroke[i][j][0])
                        prev_y = int(stroke[i][j][1])

                        # The third index [counter,2] represents the counterth end of stroke
                        # if 0 then there is still a point after it in the stroke
                        # If there is no point after it then [counter,2] is a 1 flagging it as the end of the stroke
                        if (j == (len(stroke[i])-1)):  # end of stroke
                            eos = 1
                        else:
                            eos = 0

                        stroke_data.append([x,y,eos])
            else:       #Interpolation

                for i in range(len(stroke)):
                    for j in range(len(stroke[i])):

                        # Creates each point relative to the one before it
                        # The first index [counter,0] represents the counterth x relative to previous x
                        x = int(stroke[i][j][0]) - prev_x

                        # The second index [counter,1] represents the counterth y relative to the previous y
                        y = int(stroke[i][j][1]) - prev_y


                        prev_x = int(stroke[i][j][0])
                        prev_y = int(stroke[i][j][1])


                        # The third index [counter,2] represents the counterth end of stroke
                        # if 0 then there is still a point after it in the stroke
                        # If there is no point after it then [counter,2] is a 1 flagging it as the end of the stroke
                        if (j == (len(stroke[i]) - 1)):  # end of stroke
                            eos = 1
                        else:
                            eos = 0

                        if (j == 0) or (x == 0 and y == 0):
                            stroke_data.append([x, y, eos])

                        elif (x == 0):
                            yStep = y / abs(y)
                            for k in range(abs(y)):
                                newPoint = [x, yStep, 0]
                                stroke_data.append(newPoint)
                        elif (y == 0):
                            xStep = x / abs(x)
                            for k in range(abs(x)):
                                newPoint = [xStep, y, 0]
                                stroke_data.append(newPoint)
                        else:
                            stroke_data.append([x, y, eos])

                        if (eos == 1):
                            stroke_data[-1][2] = 1

            stroke_data = np.array(stroke_data)
            return stroke_data


        # create data file from raw xml files from iam handwriting source.
        self.logger.write("\tparsing dataset...")
        
        # build the list of xml files
        strokeslist = []
        unicodelist = []
        # Set the directory you want to start from
        # ./data/lineStrokes
        # ./data/ascii

        # loops through the directory using os.walk and adds all paths to the strokeslist array
        rootDir = self.data_dir
        strokeslist = []
        unicodelist = []
        for dirName, subdirList, fileList in os.walk(rootDir):
            for fname in fileList:
                if (fname.__contains__("inkml")):
                    strokeslist.append(dirName + "/" + fname)
                elif fname.__contains__("upx"):
                    unicodelist.append(dirName + "/" + fname)
        unicodelist.sort()
        strokeslist.sort()
        # Continues the code after convert_stroke_to_array


        # build stroke database of every xml file inside iam database
        strokes = []
        unicodes = []
        for i in range(len(strokeslist)):
            stroke_file = strokeslist[i]
            unicode_file = unicodelist[i]
#                 print 'processing '+stroke_file

            unicode = getUnicode(unicode_file,self.unknowntoken) # Calls the unicode line of each respective line
            wordStrokes = convert_stroke_to_array(getStrokes(stroke_file),unicode) # calls getStrokes of the file then passes it as a parameter in convert_stroke_to_array

            strokes.append(wordStrokes)
            unicodes.append(unicode)
            #else:
                #self.logger.write("\tline length was too short. line was: " + ascii)

        #Makes sure that the number of lines (Strokes) is equal to the number of lines in the ascii       
        assert(len(strokes)==len(unicodes)), "There should be a 1:1 correspondence between stroke data and ascii labels."
        # Saves the preprocessed data as strokes_training_data.cpkl (protocol 2 stores hexa)

        f = open(data_file,"wb")
        pickle.dump([strokes,unicodes], f, protocol=2)
        f.close()
        self.logger.write("\tfinished parsing dataset. saved {} lines".format(len(strokes)))


    def dataset_analysis(self):

        dictionaryLetters = []
        charactersTobeloaded = []
        longestWord = len(max(self.raw_ascii_data, key=len))
        for i in range(1, longestWord + 1):
            dictionaryLetter, counter = self.getLettersCount(self.raw_ascii_data, i)
            dictionaryLetters.append(dictionaryLetter)
            s = '{:>20}'.format('{:<10}'.format("For")) + '{:<10}'.format(str(i)) + '{:<20}'.format("character in") \
                + '{:<10}'.format(str(counter)) + '{:<10}'.format("words") + '{:<10}'.format(str(counter * i)) \
                + '{:<10}'.format("character will be loaded\n")
            charactersTobeloaded.append(s)

        dictionary = []
        for i in range(len(dictionaryLetters)):
            line = []
            for letter in dictionaryLetters[i]:
                line.append('{:>30}'.format('{:<10}'.format("Letter")) + u'{:<10}'.format(letter) \
                            + '{:<20}'.format("Unicode") + '{:<10}'.format((letter).encode('unicode-escape')) \
                            + '{:<30}'.format("Number of repetition") + '{:<10}'.format(
                    str(dictionaryLetters[i][letter])))
            dictionary.append(line)

        f = open("logs/dataset_Info.txt", "wb")
        s = "Info of dataset \n\n"
        f.write(s)
        for i in range(len(dictionary)):
            f.write(charactersTobeloaded[i])
            f.write("\n")
            for line in dictionary[i]:
                f.write(line.encode('UTF-8') + "\n")
            f.write("\n\n\n\n")
        f.close()

    def calculate_average(self):
        average = 0
        for i in range(len(self.raw_stroke_data)):
            average += len(self.raw_stroke_data[i]) / len(self.raw_ascii_data[i].replace(" ",""))
        average = average / len(self.raw_stroke_data)
        return average

    def getLettersCount(self, data , stop):
        dictionaryLetters = {}
        counter = 0
        number = [0] * (len(self.alphabet) + 1)
        for i in range(len(data)):
            if (len(data[i]) < stop):
                continue
            counter += 1
            for j in range(len(data[i])):
                if (j > stop - 1): break
                number[self.alphabet.find(data[i][j]) + 1] += 1
        dictionaryLetters = {self.alphabet[i] : number[i + 1] for i in range(len(self.alphabet))}
        dictionaryLetters.update({'Unknown' : number[0]})
        return dictionaryLetters, counter
    # Needs optimizing, Does the first preprocessing steps and saves the file , then opens it again in load_preprocessed, does more preprocessing over here
    # without saving the data which is not optimized.

    def load_preprocessed(self, data_file):
        # Opens strokes_training_data.cpkl
        f = open(data_file,"rb")
        # Loads the contents of the file in their respective arrays
        [self.raw_stroke_data, self.raw_ascii_data] = pickle.load(f)
        f.close()

        # goes thru the list, and only keeps the text entries that have more than tsteps points
        self.stroke_data = []
        self.ascii_data = []
        self.valid_stroke_data = []
        self.valid_ascii_data = []
        # every 1 in 230 (0.4%) will be used for validation data
        cur_data_counter = 0
        validationRegex = re.compile(r"[^ "+ self.alphabet +"]")
        # print(self.calculate_average())
        # c = 0
        for i in range(len(self.raw_stroke_data)):
            data = self.raw_stroke_data[i]
            ascii = self.raw_ascii_data[i]
            for char in self.filter:
    			ascii = ascii.replace(char,"")
            # Checks if number of points > tsteps + 2 then they are valid, else ignore

            if len(data) > (self.tsteps+2) and len(ascii) > self.ascii_steps:
                # removes large gaps from the data
                # Since points are relative to each other, then if the distance between two consecutive points are large, then that means it is done by mistake
                # Self.limit = 500 , meaning points can't be further from each other than 500 pixels

                # if(data.all() != np.minimum(data, self.limit).all() or data.all() != np.maximum(data, -self.limit).all()):
                #     c += 1


                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data,dtype=np.float32)

                # Divides the x and y of each point by the data_scale (By default = 50)
                data[:,0:2] /= self.data_scale
                cur_data_counter = cur_data_counter + 1

                # Takes one of every 230 xml files and adds them to the validation set
                if cur_data_counter % 230 == 0:
                    self.valid_stroke_data.append(data)
                    ascii = validationRegex.sub("", self.raw_ascii_data[i])
                    self.valid_ascii_data.append(ascii)
                else:
                    self.stroke_data.append(data)
                    self.ascii_data.append(ascii)

        # print
        # print " number of noise",; print c
        # print
        if(self.datasetAnalysis):
            self.dataset_analysis()

        # Divides the number of lines to be studied by the batch_size (Default = 32) to make batches
        self.num_batches = int(len(self.stroke_data) / self.batch_size)
        self.logger.write("\tloaded dataset:")
        self.logger.write("\t\t{} train individual data points".format(len(self.stroke_data)))
        self.logger.write("\t\t{} valid individual data points".format(len(self.valid_stroke_data)))
        self.logger.write("\t\t{} batches".format(self.num_batches))

    def validation_data(self):
        # returns validation data
        # Returns 32 line only... out of ~ 700
        x_batch = []
        y_batch = []
        ascii_list = []
        for i in range(self.batch_size):
            # Gets the remainder, meaning 0..31
            valid_ix = i%len(self.valid_stroke_data)
            data = self.valid_stroke_data[valid_ix]
            x_batch.append(np.copy(data[:self.tsteps]))
            # Predict the next letter (Read about it in the paper)
            y_batch.append(np.copy(data[1:self.tsteps+1]))
            # Gets the ascii for each handwritten line
            ascii_list.append(self.valid_ascii_data[valid_ix])
        one_hots = [to_one_hot(s, self.ascii_steps, self.alphabet) for s in ascii_list]
        return x_batch, y_batch, ascii_list, one_hots

    def next_batch(self):
        # returns a randomized, tsteps-sized portion of the training data
        x_batch = []
        y_batch = []
        ascii_list = []
        for i in xrange(self.batch_size):
            data = self.stroke_data[self.idx_perm[self.pointer]]
            x_batch.append(np.copy(data[:self.tsteps]))
            y_batch.append(np.copy(data[1:self.tsteps+1]))
            ascii_list.append(self.ascii_data[self.idx_perm[self.pointer]])
            self.tick_batch_pointer()
        one_hots = [to_one_hot(s, self.ascii_steps, self.alphabet) for s in ascii_list]
        return x_batch, y_batch, ascii_list, one_hots

    def tick_batch_pointer(self):
        self.pointer += 1
        if (self.pointer >= len(self.stroke_data)):
            os.remove(self.idx_path)
            os.remove(self.pointer_path)
            self.reset_batch_pointer()

    def reset_batch_pointer(self):
        # Generates an array containing index of Random stroke_data
        if not (os.path.exists(self.idx_path)) :
            self.idx_perm = np.random.permutation(len(self.stroke_data))
            self.save_idx()
        else:
            self.load_idx()
        if not (os.path.exists(self.pointer_path)):
            self.pointer = 0
            self.save_pointer()
        else:
            self.load_pointer()

    def save_pointer(self):
        if(os.path.exists(self.pointer_path)):
            os.remove(self.pointer_path)
        f = open(self.pointer_path, "wb")
        pickle.dump([self.pointer], f, protocol=2)
        f.close()

    def load_pointer(self):
        f = open(self.pointer_path, "rb")
        [self.pointer] = pickle.load(f)
        f.close()

    def save_idx(self):
        if os.path.exists(self.idx_path):
            os.remove(self.idx_path)
        f = open(self.idx_path, "wb")
        pickle.dump([self.idx_perm], f, protocol=2)
        f.close()

    def load_idx(self):
        f = open(self.idx_path, "rb")
        [self.idx_perm] = pickle.load(f)
        f.close()

        

# utility function for converting input ascii characters into vectors the network can understand.
# index position 0 means "unknown"
def to_one_hot(s, ascii_steps, alphabet):

    # alphabet1=alphabet[::-1]
    # print  alphabet
    # for i in range(len(alphabet)):
    #     print alphabet[i],;print ("     "),;print i


    s = arabic_reshaper.reshape(s)


    # print  arabic_reshaper.reshape(s)
    # Sequence, gets the index of each character in the line


    steplimit=3e3; s = s[:3e3] if len(s) > 3e3 else s # clip super-long strings
    seq = [alphabet.find(char) + 1 for char in s]
    # print seq
    # for element in seq:
    #     if element == 0: continue
    #     print alphabet[element-1]
    # If number of characters > ascii steps (by default tsteps/tsteps_per_ascii (150/25 = 6))
    if len(seq) >= ascii_steps:
        # Trim the characters to limit to 6
        seq = seq[:ascii_steps]
    else:
        # If shorter, adds 0 at the end
        seq = seq + [0]*(ascii_steps - len(seq))
    # Creates a numpy matrix number of rows = ascii steps and columns length of alphabet+1 54 + 1 [In case of english]
    one_hot = np.zeros((ascii_steps,len(alphabet)+1))
    # Sets the value of the corresponding index of the character to 1 Ex. In case of B [ 0 ,0, 1 , 0 .....] (where the first index is empty as a "flag")
    one_hot[np.arange(ascii_steps),seq] = 1
    # for mat in one_hot:
    #     print mat
    #     print "\n"+"\n"
    return one_hot

def combine_image_matrixes(original, expansion):
    if len(original) == 0:
        original.append(expansion)
    else:
        additional_length = len(expansion[0])
        original = np.vstack(original)
        original_length = len(original[0])
        mod_arr = np.empty([len(original), original_length + additional_length], dtype = np.float32)
        for i in range(len(original)):
            mod_arr[i] = np.append(original[i], [0]* additional_length)
        original = mod_arr
        mod_arr = np.empty([len(expansion), original_length + additional_length], dtype = np.float32)
        for i in range(len(expansion)):
            mod_arr[i] = np.append([0] * original_length, expansion[i])
        expansion = mod_arr
        original = np.append(original, expansion, axis = 0)
    return original

def visualize(args,filename, results):
    from xml.etree.ElementTree import Element, SubElement, tostring

    if not os.path.exists(args.visual_dir):
        os.makedirs(args.visual_dir)

    save_dir_name = args.visual_dir + "/" + filename + args.preprocessing_type + ".xml"

    rootname = "root"
    root = Element(rootname)
    infosChild = SubElement(root, "infos")
    widthChild = SubElement(infosChild, "width")
    widthChild.text = "800"
    heightChild = SubElement(infosChild, "height")
    heightChild.text = "600"
    animationChild = SubElement(root, "animation")

    time = 0.01

    for i in range(0, len(results)):

        actionChild = SubElement(animationChild, "action")
        actionChild.set('time', str(time))
        startpointChild = SubElement(actionChild, "startpoint")
        startpointChild.set('x', str(results[i][0][0]))
        startpointChild.set('y', str(results[i][0][1]))
        startpointChild.set('width', "3")
        startpointChild.set('color', "255")
        startpointChild.set('alpha', "0")
        for j in range(1, len(results[i])):
            actionChild = SubElement(animationChild, "action")
            actionChild.set('time', str(time))
            pointChild = SubElement(actionChild, "point")
            pointChild.set('x', str(results[i][j][0]))
            pointChild.set('y', str(results[i][j][1]))
            time += 0.01

    tree = ET.ElementTree(root)
    tree.write(save_dir_name)

# abstraction for logging
class Logger():
    def __init__(self, args):
        self.logf = '{}train_scribe.txt'.format(args.log_dir) if args.train else '{}sample_scribe.txt'.format(args.log_dir)
        with open(self.logf, 'w') as f: f.write("Scribe: Realistic Handriting in Tensorflow\n     by Sam Greydanus\n\n\n")

    def write(self, s, print_it=True):
        if print_it:
            print s
        with open(self.logf, 'a') as f:
            f.write(s.encode("UTF-8") + "\n")
