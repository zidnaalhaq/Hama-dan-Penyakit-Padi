from sklearn.preprocessing import LabelEncoder
import pandas as pd
from flask import Flask, render_template, request, make_response, json, url_for
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

app = Flask(__name__)


def ciriA(agrumentA):
    match agrumentA:
        case 0:
            return "Daun tergulung"
        case 1:
            return "Tersisa tulang tulang daun"
        case 2:
            return "Terbentuk gelembung-gelembung pada daun"
        case 3:
            return "Bibit berubah warna menjadi kekuningan"
        case 4:
            return "Tanaman menjadi layu"
        case 5:
            return "Daun terdapat bercak-bercak bekas isapan"
        case 6:
            return "Tanaman Roboh"
        case 7:
            return "Tangkai patah"


def ciriB(agrumentB):
    match agrumentB:
        case 0:
            return "Daun berubah warna menjadi kuning"
        case 1:
            return "Tampak bekas keratan ulat"
        case 2:
            return "Tunas tidak dapat berbunga"
        case 3:
            return "Bibit menjadi layu dan"
        case 4:
            return "Daun mengalami perubahan warna"
        case 5:
            return "Buah terdapat bintik-bintik hitam bekas tusukan hama"
        case 6:
            return "Pucuk mudah dicabut"
        case 7:
            return "Terdapat bekas keratan tikus"
        case 8:
            return "Buah mudah rontok"
        case 9:
            return "Daun sobek"


def ciriC(agrumentC):
    match agrumentC:
        case 0:
            return "Beberapa gabah tidak berisi"
        case 1:
            return "Terlihat potongan tangkai mati"
        case 2:
            return "Warnanya hijau kekuningan"
        case 3:
            return "Bibit Mudah dicabut"
        case 4:
            return "Tanaman mudah Roboh"
        case 5:
            return "Buah hampa"
        case 6:
            return "Daun menjadi kering"
        case 7:
            return "Padi yang masih muda tampak botak"
        case 8:
            return "Burung-burung pemakan padi berkeliaran disekitar tananaman padi"
        case 9:
            return "Daun habis tinggal tulang"


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/main")
def aboutUs():
    return render_template("main.html")


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        ciri1 = request.form["ciri1"]
        ciri2 = request.form["ciri2"]
        ciri3 = request.form["ciri3"]
        print(ciri1)
        # convert obj to int
        agrumentA = int(ciri1)
        print(type(agrumentA))
        agrumentB = int(ciri2)
        print(type(agrumentB))
        agrumentC = int(ciri3)
        print(type(agrumentC))

        agrumentA = ciriA(agrumentA)
        print(ciriA(agrumentA))
        agrumentB = ciriB(agrumentB)
        print(ciriA(agrumentB))
        agrumentC = ciriC(agrumentC)
        print(ciriA(agrumentC))

        # data test
        Xtes = np.array([ciri1, ciri2, ciri3])
        print(Xtes)

        # import csv
        dataset = pd.read_csv(open("data/dataset.csv", "r"))
        print(dataset)
        feature_names = dataset.columns[:3]
        target_names = dataset["hasil"].unique().tolist()
        # labeling
        le = LabelEncoder()
        #
        for column in dataset:
            if dataset[column].dtypes == object:
                dataset[column] = le.fit_transform(dataset[column])
        print(dataset)
        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        model = tree.DecisionTreeClassifier(
            random_state=0,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0,
            max_leaf_nodes=None,
            min_impurity_decrease=0,
        )

        clf = model.fit(x, y)
        plt.figure(figsize=(40, 25))
        plot_tree(
            model,
            feature_names=feature_names,
            class_names=target_names,
            filled=True,
            rounded=True,
        )

        plt.savefig("tree_visualization.png")
        hasil_prediksi = clf.predict([Xtes])

        if hasil_prediksi == [0]:
            agrumentA = agrumentA
            agrumentB = agrumentB
            agrumentC = agrumentC
            hasil = "Trip Padi (Wereng)"
            keterangan = "Trip padi adalah"
            penanganan = "Gunakan varietas tahan wereng coklat, seperti: Ciherang, Kalimas, Bondoyudo, Sintanur dan Batang Gadis. Berikan pupuk K untuk mengurangi. Monitor pertanaman paling lambat 2 minggu. Bila populasi hama di bawah ambang ekonomi gunakan insektisida botani atau jamur ento-mopatogenik. Bila populasi hama di atas ambang ekonomi gunakan insektisida kimiawi yang direkomendasi."
            gambar = "penyakit1.jpg"
        elif hasil_prediksi == [1]:
            agrumentA = agrumentA
            agrumentB = agrumentB
            agrumentC = agrumentC
            hasil = "Ulat Tentara"
            keterangan = "penyakit dua adalah"
            penanganan = "Pengamatan rutin dilakukan setiap waktu sehingga kita harus waspada dan paling lama seminggu sekali. Kita amati hama apa saja pada tanaman kita serta kita amati gejala-gejala serangan hama maupun penyakit yang ada. Dengan pengamatan secara dini otomatis kita akan cepat mendeteksi adanya serangan hama ulat grayak. Kalau kita mengetahui secara awal serangan hama tersebut maka dengan mudah kita bisa mengendalikan hama tersebut; Apabila dalam pengamatan ditemukan intensitas serangannya masih sangat ringan atau rendah kita bisa melakukan tindakan pengendalian secara mekanik, yaitu mengambil telur, ulat maupun kepompong ulat grayak tersebut; Apabila dalam pengamatan didapatkan ulat grayak sebanyak 2 ekor untuk tiap rumpun padi, maka pengendalian harus segera dilakukan dengan menggunakan insektisida sintetik agar serangan hama tidak meluas; Selain menggunakan insektisida sintetik kita juga bisa menggunakan insektisida biologis yang lebih ramah terhadap lingkungan (Beauveria bassiana). Selain itu kita juga bisa memanfaatkan bahan-bahan tanaman di sekitar kita untuk dijadikan pestisida nabati seperti daun mimba, akar tuba, jengkol dll."
            gambar = "penyakit2.jpg"
        elif hasil_prediksi == [2]:
            agrumentA = agrumentA
            agrumentB = agrumentB
            agrumentC = agrumentC
            hasil = "Hama Ganjur"
            keterangan = "penyakit tiga adalah"
            penanganan = "Atur waktu tanam agar puncak curah hujan tidak bersamaan dengan stadia vegetatif. Bajak ratun/tunggal dari tanaman sebelumnya dan buang/bersihkan semua tanaman inang akternatif selama masa bera, seperti padi liar Oryza rufipogon untuk mengurangi infestasi hama. Tanam varietas tahan. Hama ganjur dewasa sangat tertarik terhadap cahaya, oleh karena itu lampu perangkap dapat digunakan untuk menangkap hama ganjur dewasa. Insektisida glanular yang berbahan aktif karbofuran dapat digunakan karena bekerja secara sistematik."
            gambar = "penyakit3.jpg"
        elif hasil_prediksi == [3]:
            agrumentA = agrumentA
            agrumentB = agrumentB
            agrumentC = agrumentC
            hasil = "Hama Uret atau Kuuk"
            keterangan = "penyakit empat adalah"
            penanganan = "Pengendalian hama uret telah dilakukan melalui berbagai cara seperti kultur teknis (tanam serempak, rotasi tanaman dengan tanaman bukan inang, sanitasi lahan, pengolahan lahan yang dalam), pengendalian biologis dengan jamur Metarhizium anisopliae, pengendalian secara mekanik (mengumpulkan uret pada saat pengolahan tanah, menangkap imago dengan memasang lampu perangkap), dan pengendalian secara kimia dengan aplikasi karbofuran 20 kg/ha secara tugal pada saat tanam. Pengendalian secara kimia, selain dengan aplikasi karbofuran 20 kg/ha, saat ini telah diperoleh teknik pengendalian yang efektif yang mampu menekan serangan hama uret atau lundi pada pertanaman padi gogo dengan teknik seed treatment."
            gambar = "penyakit4.jpg"
        elif hasil_prediksi == [4]:
            agrumentA = agrumentA
            agrumentB = agrumentB
            agrumentC = agrumentC
            hasil = "Hama Orong-Orong (Anjing Tanah)"
            keterangan = "penyakit lima adalah"
            penanganan = "Usahakan membuat lahan persawahan rata dengan tujuan agar air dapat menggenang secara merata sehingga dapat memperkecil serangan hama orong-orong yang diharapkan nantinya mengurangi kerusakan pada tanaman padi. Lakukan penggenangan sawah 3 – 4 hari agar dapat mematikan telur orong-orong yang berada di dalam tanah Lakukan pencelupan bibit padi dalam larutan insektisida sebelum dilakukan penanaman agar terhindar dari serangan orong-orong. Melakukan pengumpanan dengan memanfaatkan sekam padi yang telah di campur dengan insektesida. Letakkan sekam tersebut di beberapa tiitk pada lokasi lahan pertanian. Menggunakan cara ini di anggap ramah lingkungan dan aman meskipun melibatkan insektisida kimia. Cara ini dapat diaplikasikan kepada berbagai jensi tanaman seperti palawija. Penggunaan insektisida (bila diperlukan) yang berbahan aktif carbufuron atau fipronil Pengendalian hama orong-orong secara alami dengan pestisida nabati menggunakan kulit jengkol, tembakau dan akar tuba."
            gambar = "penyakit5.jpg"
        elif hasil_prediksi == [5]:
            agrumentA = agrumentA
            agrumentB = agrumentB
            agrumentC = agrumentC
            hasil = "Walang Sangit"
            keterangan = "penyakit enam adalah"
            penanganan = "Kendalikan gulma di sawah dan di sekitar pertanaman. Pupuk lahan secara merata agar pertumbuhan tanaman seragam. Tangkap walang sengit dengan menggunakan 15arring sebelum stadia pembungaan. Umpan walang sangit dengan menggunakan ikan yang sudah busuk, daging yang  sudah rusak, atau dengan kotoran ayam. Apabila serangan sudah mencapai ambang  ekonomi, lakukan penyemprotan insektisida. Lakukan penyemprotan pada pagi sekali atau sore hari ketika walang sangit berada di kanopi. "
            gambar = "penyakit6.jpg"
        elif hasil_prediksi == [6]:
            agrumentA = agrumentA
            agrumentB = agrumentB
            agrumentC = agrumentC
            hasil = "Penggerek Batang Padi"
            keterangan = "penyakit tujuh adalah"
            penanganan = "Bila populasi tinggi (di atas ambang ekonomi) aplikasikan insektisida. Bila genangan air dangkal aplikasikan insektisida butiran seperti karbofuran dan fipronil, dan bila genangan air tinggi aplikasikan insektsida cair seperti dimehipo, bensultap, amitraz dan fipronil."
            gambar = "penyakit7.jpg"
        elif hasil_prediksi == [7]:
            agrumentA = agrumentA
            agrumentB = agrumentB
            agrumentC = agrumentC
            hasil = "Hama Tikus"
            keterangan = "penyakit delapan adalah"
            penanganan = "Pengendalian hama tikus terpadu (PHTT) didasarkan  pada pemahaman ekologi jenis tikus, dilakukan secara dini, intensif dan terus menerus dengan  memanfaatkan teknologi pengendalian yang sesuai dan tepat waktu. Pengendalian tikus ditekankan  pada  awal  musim tanam untuk menekan populasi awal tikus sejak awal pertanaman sebelum tikus memasuki masa reproduksi. Kegiatan tersebut  meliputi gropyok masal, sanitasi habitat, pemasangan TBS (Trap Barrier System) dan LTBS (Linier Trap Barrier System)."
            gambar = "penyakit8.jpg"
        elif hasil_prediksi == [8]:
            agrumentA = agrumentA
            agrumentB = agrumentB
            agrumentC = agrumentC
            hasil = "Hama Burung"
            keterangan = "penyakit sembilan adalah"
            penanganan = "Menanam tanaman berwarna mencolok. Pada umumnya burung pipit tidak menyukai warna-warna yang mencolok, seperti warna kuning. Oleh karena itu, petani bisa menghalau serangan hama ini dengan menanam bunga matahari atau tahi ayam. Bunga tersebut dapat ditanam di pematang sawah sebagai pembatas (border). Dengan begitu, burung akan enggan mendekat ke tanaman padi. Memasang benda-benda mengkilap. Sama seperti warna mencolok, burung pipit juga tidak suka dengan benda-benda yang mengilap. Oleh karena itu, hama burung bisa diusir dengan menggunakan benda-benda yang mengilap, seperti plastik atau bekas piringan cakram (CD audio/video). Memasang jaring pernagkap. Pengendalian hama burung dapat dilakukan dengan menggunakan jaring khusus untuk menangkap burung, petani biasanya menggunakan jaring bekas menangkap ikan. Jaring tersebut ditancapkan pada beberapa kayu atau bambu di pematang sawah. Namun, cara ini membutuhkan biaya yang besar karena petani membutuhkan banyak jaring untuk melindungi lahan sawah yang cukup besar. Jengkol, aroma jengkol yang tidak disukai oleh burung pipit dimanfaatkan oleh petani untuk mengusir hama tersebut. Cara penggunaannya cukup mudah, petani akan merendam jengkol selama beberapa hari hingga air rendamannya mengeluarkan aroma jengkol yang menyengat. Setelah itu, air rendaman jengkol tersebut akan dimasukkan ke botol, kemudian botol tersebut diletakkan di beberapa sudut sawah atau disemprot ke tanaman padi."
            gambar = "penyakit9.jpg"
        elif hasil_prediksi == [9]:
            agrumentA = agrumentA
            agrumentB = agrumentB
            agrumentC = agrumentC
            hasil = "Penggulung dan Pelipat Daun"
            keterangan = "penyakit sepuluh adalah"
            penanganan = "Bersihkan lahan dari rumput/gulma yang menjadi makanan alternatifnya  Pada areal yang sempit, potong/gunting daun yang terkena hama Gunakan insektisida bila sudah mencapai ambang ekonomiac."
            gambar = "penyakit10.jpg"
        else:
            agrumentA = agrumentA
            agrumentB = agrumentB
            agrumentC = agrumentC
            hasil = "Penyakit Tidak Ada"
            penanganan = ""
            keterangan = "Tidak Ada"
        print(hasil_prediksi)
        return render_template(
            "result.html",
            hasil=hasil,
            keterangan=keterangan,
            penanganan=penanganan,
            gambar=gambar,
            ciri1=agrumentA,
            ciri2=agrumentB,
            ciri3=agrumentC,
        )


if __name__ == "__main__":
    app.run(debug=True)
