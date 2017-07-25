import numpy as np
import math
import random
import os
import cPickle as pickle
import xml.etree.ElementTree as ET

from utils import *

class DataLoader():
    def __init__(self, args, logger, limit = 500):
        self.data_dir = args.data_dir
        self.alphabet = args.alphabet
        self.batch_size = args.batch_size
        self.tsteps = args.tsteps
        self.data_scale = args.data_scale # scale data down by this factor
        self.ascii_steps = args.tsteps/args.tsteps_per_ascii
        self.logger = logger
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
        # create data file from raw xml files from iam handwriting source.
        self.logger.write("\tparsing dataset...")
        
        # build the list of xml files
        filelist = []
        # Set the directory you want to start from
        # ./data/lineStrokes
        # ./data/ascii

        # loops through the directory using os.walk and adds all paths to the filelist array
        rootDir = stroke_dir
        for dirName, subdirList, fileList in os.walk(rootDir):
            for fname in fileList:
                filelist.append(dirName+"/"+fname)
        # Continues the code after convert_stroke_to_array

        # function to read each individual xml file
        def getStrokes(filename):
            # Each XML File represents an entire line in the form
            # Uses an XML Parser that creates a tree of the xml file
            tree = ET.parse(filename)
            root = tree.getroot()

            result = []

            x_offset = 1e20
            y_offset = 1e20

            # In the XML file loops through the "<WhiteboardDescription>" which stores the bounding box of the input
            # then it goes ahead and gets the minimum x and minimum y to use it later on

            for i in range(1, 4):
                x_offset = min(x_offset, float(root[0][i].attrib['x']))
                y_offset = min(y_offset, float(root[0][i].attrib['y']))

            # Createss a padding
            x_offset -= 100
            y_offset -= 100


            # Finds the "Stroke" tag
            for stroke in root[1].findall('Stroke'):
                points = []
                # Loops through the Points tags
                for point in stroke.findall('Point'):
                    # Adds the points relative to the 0,0 [So text is consistent]
                    points.append([float(point.attrib['x'])-x_offset,float(point.attrib['y'])-y_offset])
                result.append(points)

            # result is basically a 2D array each outer array represents a Stroke and each inner array represents a stroke
            # result[0][0] represents the first point in the first stroke
            # after that passes the result to convert_stokes_to_array
            return result

        
        # function to read each individual xml file
        def getAscii(filename, line_number):
            # Gets the CSR ascii content
            with open(filename, "r") as f:
                s = f.read()
            s = s[s.find("CSR"):]
            if len(s.split("\n")) > line_number+2:
                s = s.split("\n")[line_number+2]
                return s
            else:
                return ""
                
        # converts a list of arrays into a 2d numpy int16 array
        def convert_stroke_to_array(stroke):
            n_point = 0
            for i in range(len(stroke)):
                n_point += len(stroke[i])
            #Creates a numpy matrix of n columns and 3 rows
            stroke_data = np.zeros((n_point, 3), dtype=np.int16)

            prev_x = 0
            prev_y = 0
            counter = 0

            for j in range(len(stroke)):
                for k in range(len(stroke[j])):

                    # Creates each point relative to the one before it
                    # The first index [counter,0] represents the counterth x relative to previous x
                    stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x

                    # The second index [counter,1] represents the counterth y relative to the previous y
                    stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y

                    prev_x = int(stroke[j][k][0])
                    prev_y = int(stroke[j][k][1])

                    # The third index [counter,2] represents the counterth end of stroke
                    # if 0 then there is still a point after it in the stroke
                    stroke_data[counter, 2] = 0

                    # If there is no point after it then [counter,2] is a 1 flagging it as the end of the stroke
                    if (k == (len(stroke[j])-1)): # end of stroke
                        stroke_data[counter, 2] = 1
                    counter += 1
            return stroke_data

        # build stroke database of every xml file inside iam database
        strokes = []
        asciis = []
        for i in range(len(filelist)):
            if (filelist[i][-3:] == 'xml'):  # Checks the extension of the file if its xml then it means that it contains strokes and points
                stroke_file = filelist[i]
#                 print 'processing '+stroke_file
                stroke = convert_stroke_to_array(getStrokes(stroke_file)) # calls getStrokes of the file then passes it as a parameter in convert_stroke_to_array

                #Gets the corresponding ascii file for the lineStroke xml
                #Changes the directory name from lineStrokes to ascii and removes some of the subfolder paths that aren't needed
                ascii_file = stroke_file.replace("lineStrokes","ascii")[:-7] + ".txt"

                # Gets the line_number depending on the line number in the stroke_file name
                # Example: a01-000u-01.xml , gets the 01 at the end to represent the line number
                line_number = stroke_file[-6:-4]

                # Removes one from the line_number (So index starts from 0)
                line_number = int(line_number) - 1
                ascii = getAscii(ascii_file, line_number) # Calls the ascii line of each respective line
                # Checks the length of the line, if its greater than 10 characters then it is okay.
                # Assumption : First commit had default tsteps of 250 , and since ascii = tsteps/tsteps_per_ascii 250/25 = 10
                # Why 10?
                if len(ascii) > 10:
                    strokes.append(stroke)
                    asciis.append(ascii)
                else:
                    self.logger.write("\tline length was too short. line was: " + ascii)

        #Makes sure that the number of lines (Strokes) is equal to the number of lines in the ascii       
        assert(len(strokes)==len(asciis)), "There should be a 1:1 correspondence between stroke data and ascii labels."
        # Saves the preprocessed data as strokes_training_data.cpkl (protocol 2 stores hexa)
        f = open(data_file,"wb")
        pickle.dump([strokes,asciis], f, protocol=2)
        f.close()
        self.logger.write("\tfinished parsing dataset. saved {} lines".format(len(strokes)))

    def calculate_average(self):
        average = 0
        for i in range(len(self.raw_stroke_data)):
            average += len(self.raw_stroke_data[i]) / len(self.raw_ascii_data[i].replace(" ",""))
        average = average / len(self.raw_stroke_data)
        return average
    def getLettersCount(self, data):
        dictionaryLetters = {}
        number = [0] * (len(self.alphabet) + 1)
        for i in range(len(data)):
            for j in range(len(data[i])):
                number[self.alphabet.find(data[i][j]) + 1] += 1
        dictionaryLetters = {self.alphabet[i] : number[i + 1] for i in range(len(self.alphabet))}
        dictionaryLetters.update({'Unknown' : number[0]})
        print dictionaryLetters
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
        # every 1 in 20 (5%) will be used for validation data
        cur_data_counter = 0
        # print(self.calculate_average())
        for i in range(len(self.raw_stroke_data)):
            data = self.raw_stroke_data[i]
            ascii = self.raw_ascii_data[i].replace(" ","")
            # Checks if number of points > tsteps + 2 then they are valid, else ignore
            if len(data) > (self.tsteps+2) and len(ascii) > self.ascii_steps:
                # removes large gaps from the data
                # Since points are relative to each other, then if the distance between two consecutive points are large, then that means it is done by mistake
                # Self.limit = 500 , meaning points can't be further from each other than 500 pixels
                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data,dtype=np.float32)

                # Divides the x and y of each point by the data_scale (By default = 50)
                data[:,0:2] /= self.data_scale
                cur_data_counter = cur_data_counter + 1

                # Takes one of every 20 xml files and adds them to the validation set
                if cur_data_counter % 20 == 0:
                  self.valid_stroke_data.append(data)
                  self.valid_ascii_data.append(ascii)
                else:
                    self.stroke_data.append(data)
                    self.ascii_data.append(ascii)

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
    steplimit=3e3; s = s[:3e3] if len(s) > 3e3 else s # clip super-long strings
    # Sequence, gets the index of each character in the line
    seq = [alphabet.find(char) + 1 for char in s]
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

# abstraction for logging
class Logger():
    def __init__(self, args):
        self.logf = '{}train_scribe.txt'.format(args.log_dir) if args.train else '{}sample_scribe.txt'.format(args.log_dir)
        with open(self.logf, 'w') as f: f.write("Scribe: Realistic Handriting in Tensorflow\n     by Sam Greydanus\n\n\n")

    def write(self, s, print_it=True):
        if print_it:
            print s
        with open(self.logf, 'a') as f:
            f.write(s + "\n")
