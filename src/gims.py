# Laden der verschiedenen Bibs
import cv2
import numpy as np
import pywt
import nibabel as nib
from pathlib import Path


#def concat(a, b):
#    return int(f"{a}{b}")

class Preprocessor:
    def __init__(self):
        """
        Funktionen:
            CreateFolderXRayImages --> Das schärfste Bild eines *.rec Files wird in einem Ordner gespeichert
            SearchForGoldenImageWF --> Vorsortierung der *.rec mit der Wavelet Transformation
            ohne Referenzbilder | 100% Trefferqoute
        ___________________________________________________________________________________________________________
         Input:
        """
    def GetRecfileDimensions(self, base_dir):
        # Globale Klassenvariablen
        self.base_dir = base_dir

        idx = base_dir + ".vgi"
        # Auslesen des .vgi File | Size wird benötigt
        sizelist = list()
        term = "size"
        file = open(idx, "r")
        # Jede Zeile der Textdatei wird nach size durchsucht
        for line in file:
            line.strip().split('/n')
            # Wenn die Zeile : size gefunden wird, dann wird diese abgespeichert | Loop wird beendet
            if line.startswith('size'):
                sizelist.append(line)
                break
        file.close()
        # Auslesen und abspeichern der Zeile size
        gleich = False
        position = list()
        count = 0
        # Loopen über das Array sizelist | r := Position im Array und e:= ist der Buchstabe oder die
        # Zahl im Array
        # Gesucht wird das "=" und alle "Leerzeichen" danach
        for r, e in enumerate(sizelist[0]):
            # Position des "="
            if e == "=":
                gleich = True
                position.append(r)
            # Position der "Leerzeichen" nach dem "="
            if gleich and e == " ":
                # Zählen der Anzahl der "Leerzeichen"
                count += 1
                position.append(r)
        # Speichern der letzten Position nach dem Loop
        last = r
        # Wenn size = 45 45 45 | 3 Leerzeichen
        if count == 3:
            position = position[1:]
        # Wenn size =45 45 45 | 2 Leerzeichen
        else:
            count = 3
        position.append(last)
        # Abspeichern der einzelnen Dimensionen von string zu int
        w, h, z = str(""), str(""), str("")
        countsec = 0
        for i, r in enumerate(position):
            for k in range(position[i] + 1, position[i + 1]):
                if countsec == 0:
                    w = w + str(sizelist[0][k])
                if countsec == 1:
                    h = h + str(sizelist[0][k])
                if countsec == 2:
                    z = z + str(sizelist[0][k])
            countsec += 1
            if countsec == count:
                break
        w = int(w)
        h = int(h)
        z = int(z)

        return w, h, z

    def concat(self, a, b):
        return int(f"{a}{b}")

    def CreateAndReturnGims(self, base_dir, side, boardtype, cutArray, jump=None, level=3, number_pics=1):
        """
        Start function fir wavelet, reads image and metadata seperatetly and merges it together as one image stack.
        Calls the waveletfunctions with specific options.

        :param base_dir: Path to rec and vgi files without fie ending
        :param Jump=None: Überspringen bestimmter *rec-Files
        :param level=3: Mehrfachauflösungsanalyse
        :param number_pics=1: Anzahl der zu speichernden Bildern | Funktion ist noch nicht ausgereift
        """

        ## Globale Klassenvariablen
        #self.base_dir = base_dir

        # Erstellen Pfade für *rec-File
        uri = base_dir + ".rec"

        # Öffnen des .rec File
        img = open(uri, "rb")
        scene_image_array = np.fromfile(img, dtype=np.uint8)

        w, h, z = self.GetRecfileDimensions(base_dir)

        # Übergabe der Dimensionen aus dem .vgi File
        Dim_size = np.array((z, h, w), dtype=np.int)
        # Reshapen des .rec Files in die jeweiligen Dimensionen
        # Improved Wavelet algorithm cutting the X1 arrays before doing the wavelet transformation
        if side == 'X2' and (boardtype == 'A5E3737765004' or boardtype == 'A5E3737764905'):
            # the stack of images will not be cut, it will analyzed in one go
            img_arr = scene_image_array.reshape(Dim_size[0], Dim_size[1], Dim_size[2])
            GI, Med = self.SearchForGoldenImageWFDekombi(img_arr, level=level, AnzahlBilder=number_pics)
            return GI[0], Med[0], w, h

        elif side == 'X1' and cutArray is True and (boardtype == 'A5E3737765004' or boardtype =='A5E3737764905'):
            img_arr = scene_image_array.reshape(Dim_size[0], Dim_size[1], Dim_size[2])
            img_arr_top = img_arr[:z, :290, :w]
            img_arr_bottom = img_arr[:z, 290:, :w]

            GI_top, Med_top = self.SearchForGoldenImageWFDekombi(img_arr_top, level=level, AnzahlBilder=number_pics)
            GI_bottom, Med_bottom = self.SearchForGoldenImageWFDekombi(img_arr_bottom, level=level, AnzahlBilder=number_pics)
            GI = np.concatenate((GI_top[0], GI_bottom[0]), axis=0)
            Med = self.concat( Med_top[0], Med_bottom[0])
            return GI, Med, w, h

        else:
            #print('X1 und False')
            img_arr = scene_image_array.reshape(Dim_size[0], Dim_size[1], Dim_size[2])
            GI, Med = self.SearchForGoldenImageWFDekombi(img_arr, level=level, AnzahlBilder=number_pics)
            return GI[0], Med[0], w, h


    def SearchForGoldenImageWFDekombi(self, imagenp, level=3, AnzahlBilder=1):
        """
        Vorsortierung der *.rec mit der Wavelet Transformation ohne Referenzbilder | 100% Trefferqoute
        Input:
            PathRecFiles --> Übergeordneter Ordnerpfad zu den rec Files |
            Ordnername in dem der Ordner AusgeleseneBilder liegt
            level=3 --> Level der Dekombisation
            AnzahlBilder=1 --> Anzahl zu speichernder Bilder eines *rec-Files
        """

        # Öffnen der einzelnen Bilder |  X-Ray-Images --> VT-X750-0099_0000000095_0_C-M3258839_20200304112248
        # --> Volume_(X1)_0000000015_0000024484 --> ... 00.jpg

        SaveAllSdd3 = list()
        for img in imagenp:
            # Überprüfen, ob eine GrayImage vorliegt
            shapeIt = img.shape
            if pywt.dwtn_max_level((shapeIt[0], shapeIt[1]), 'bior1.5') < level:
                level = pywt.dwtn_max_level((shapeIt[0], shapeIt[1]), 'bior1.5')
            # Wenn kein GreyImage vorhanden, dann umwandeln
            if len(shapeIt) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Anwendung der Wavlet Transformation (WT)
            coeffs2O = pywt.wavedec2(img, 'bior1.5', mode='periodization', level=level)

            SaveAllSdd = list()
            # Summieren der Wavelet-Koeffizienten
            for split in coeffs2O[1:]:
                LH = np.sum(abs(split[0]))
                LO = np.sum(abs(split[1]))
                HH = np.sum(abs(split[2]))
                Sum = LH + LO
                SaveAllSdd.append(Sum)
            SaveAllSdd3.append(SaveAllSdd)
        SaveAllSdd3 = np.array(SaveAllSdd3)
        median = list()

        # Sortieren der einzelnen Bild Position absteigend nach der Summe der Wavelet-Koeffizienten
        for ii in range(level):
            BestofThreeFront1 = sorted(range(len(SaveAllSdd3[:, ii])), key=lambda i: SaveAllSdd3[:, ii][i],
                                       reverse=False)[-1:]
            # Abspeichern der Position mit der höchsten Summe
            median.append(BestofThreeFront1)
        # Bilden des Medians der jeweils höchsten Position über alle Auflösungslevels
        GoldList = list()
        MedianList = list()
        Median = int(np.median(median))
        GoldImgNP = imagenp[Median]
        GoldList.append(GoldImgNP)
        MedianList.append(int(Median))

        return GoldList, MedianList

    def RecToNii(self, base_dir, filename):
        uri = base_dir + ".rec"
        img = open(uri, 'rb')

        w, h, z = self.GetRecfileDimensions(base_dir)

        flat_image_array = np.fromfile(img, dtype=np.uint8)
        img_dimensions = np.array((z, h, w), dtype=int)
        image_array = flat_image_array.reshape(img_dimensions[0], img_dimensions[1], img_dimensions[2])
        nii_file = nib.Nifti1Image(image_array, affine=np.eye(4))
        nii_filename = str(Path(filename).stem)
        nib.save(nii_file, ('niiCache/' + nii_filename))
        #return nii_file


    # Ordnerpfad der *rec Files

# Example Excution
# base_dir = r"D:\VT-X750-0099_0000000078_0_C-MF414732_20200623171204"
# # Endordnerpfad der schärfsten Bilder
# transfair_dir = r"D:\GIMS"
# # Skript zur Ausführung
# Array = PushData().CreateFolderXRayImages(path_to_rec*, jump="X2", DeleteOldStuff=True, level=3,
#                                           number_pics=1)
