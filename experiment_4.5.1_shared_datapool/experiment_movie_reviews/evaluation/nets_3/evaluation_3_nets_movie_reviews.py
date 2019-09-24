import numpy as np
import matplotlib.pyplot as plt

VERLAUF_1_TESTSET_ACCURACY = [0.6771993178854449, 0.7128097101013141, 0.7326712809710101, 0.7435048650817534, 0.7513291202728458, 0.7584511987160196, 0.7648711004112749, 0.7663757648711004, 0.7696860266827165, 0.7756043735580299, 0.7755040625940415, 0.7823252081452503, 0.7842311164610292, 0.7862373357407965, 0.7865382686327616, 0.78724044538068, 0.7883438659845521, 0.785535158992878, 0.7861370247768081, 0.792456615508075]
VERLAUF_1_TESTSET_PRECISION = [0.6704018547140649, 0.6979263964132262, 0.7226600038439361, 0.7312607285905016, 0.7498502695148732, 0.7503891050583658, 0.7541425818882467, 0.7626212631162146, 0.7689227072486985, 0.7611798287345385, 0.7678641155798516, 0.78705594120049, 0.7894091187896136, 0.7796743182264076, 0.7965502909393184, 0.7935563308023804, 0.8121739130434783, 0.78125, 0.785, 0.799260324635299]
VERLAUF_1_TESTSET_RECALL = [0.6965074267362505, 0.7498996386993175, 0.7547169811320755, 0.7695704536330791, 0.7539140907266159, 0.7741870734644721, 0.7856282617422722, 0.7731834604576475, 0.7707747892412685, 0.8028904054596547, 0.7894419911682056, 0.7737856282617422, 0.7749899638699318, 0.797671617824167, 0.7693697310317141, 0.7761942994781212, 0.7498996386993175, 0.7928542753914091, 0.7878362103572862, 0.7808109193095143]
VERLAUF_1_TESTSET_F1 = [0.6832053553849183, 0.7229801644895986, 0.7383406971035837, 0.7499266503667481, 0.7518766890201182, 0.7621023513139695, 0.7695635076681085, 0.767866042061198, 0.7698476343223737, 0.7814789489108137, 0.7785035629453682, 0.7803643724696356, 0.7821330902461259, 0.7885702946720904, 0.7827241168062079, 0.784779299847793, 0.7797954498017117, 0.7870093644152223, 0.7864155479863754, 0.7899279114630927]

VERLAUF_2_TESTSET_ACCURACY = [0.6765974521015147, 0.7128097101013141, 0.7333734577189287, 0.7441067308656836, 0.7509278764168924, 0.7587521316079847, 0.7638679907713913, 0.7673788745109841, 0.7685826060788444, 0.7755040625940415, 0.7761059283779718, 0.7806199217574481, 0.7826261410372154, 0.7842311164610292, 0.7848329822449593, 0.7843314274250176, 0.7875413782726453, 0.784732671280971, 0.7845320493529943, 0.786939512488715]
VERLAUF_2_TESTSET_PRECISION = [0.669822256568779, 0.69822263797942, 0.7232897770945427, 0.7319213890478916, 0.7493514268609061, 0.7495642068564788, 0.7531791907514451, 0.7623645320197044, 0.7664873480773062, 0.7599469496021221, 0.7686596326690114, 0.7877290508544369, 0.7897879349392629, 0.7741623087352315, 0.7939896373056995, 0.7914779744750926, 0.811439756415833, 0.7846246487354476, 0.7855703345425232, 0.7668035847647499]
VERLAUF_2_TESTSET_RECALL = [0.6959052589321557, 0.7490967482938579, 0.7555198715375351, 0.769971898835809, 0.7537133681252509, 0.776796467282216, 0.7846246487354476, 0.776595744680851, 0.7721798474508229, 0.8050983540746688, 0.7896427137695704, 0.7679646728221597, 0.769971898835809, 0.80228823765556, 0.7689682858289844, 0.7717784022480931, 0.748896025692493, 0.7846246487354476, 0.7824167001204335, 0.8243677238057006]
VERLAUF_2_TESTSET_F1 = [0.682614687930695, 0.7227655659920595, 0.7390536029844884, 0.7504646385601096, 0.7515260682477735, 0.7629374075899458, 0.7685804168305151, 0.7694143382718505, 0.7693230676932307, 0.7818713450292398, 0.779009900990099, 0.7777213131415793, 0.7797540400447202, 0.7879743716116313, 0.7812786784949527, 0.7815040650406504, 0.7789144050104385, 0.7846246487354476, 0.7839903459372486, 0.7945443993035404]

VERLAUF_3_TESTSET_ACCURACY = [0.6769986959574682, 0.7146153074531046, 0.7326712809710101, 0.7448089076136022, 0.7511284983448691, 0.7576487110041128, 0.7633664359514495, 0.7677801183669375, 0.7686829170428328, 0.7757046845220182, 0.778112147657739, 0.7790149463336342, 0.7810211656134015, 0.7848329822449593, 0.7851339151369244, 0.7849332932089478, 0.7876416892366336, 0.7874410673086568, 0.7851339151369244, 0.7877420002006219]
VERLAUF_3_TESTSET_PRECISION = [0.6713341112407624, 0.6997569639184894, 0.7231747254864188, 0.7326335877862595, 0.7485589346054462, 0.74902950310559, 0.75323421509944, 0.7616244849911713, 0.7651605231866825, 0.7611259033853176, 0.7689320388349514, 0.7869089407392112, 0.7924764890282132, 0.7727360123053258, 0.7981940361192776, 0.793910521955261, 0.8102664067576348, 0.7835214894038424, 0.7806324110671937, 0.7743108728943339]
VERLAUF_3_TESTSET_RECALL = [0.6928944199116821, 0.7513046969088719, 0.753512645523886, 0.7705740666399037, 0.755921316740265, 0.7745885186672019, 0.7830188679245284, 0.779205138498595, 0.7749899638699318, 0.8032918506623846, 0.7948615014050582, 0.764953833801686, 0.7611401043757527, 0.8067041348855881, 0.7629466077880369, 0.7693697310317141, 0.7509032517061421, 0.7940586109995985, 0.7928542753914091, 0.8119229225210759]
VERLAUF_3_TESTSET_F1 = [0.6819438956934019, 0.7246152356983835, 0.7380320456109309, 0.7511250244570534, 0.752222111255368, 0.7615946319321097, 0.7678378112390513, 0.7703145153289017, 0.7700438771439968, 0.781640625, 0.7816818002368733, 0.7757760814249364, 0.7764922698883998, 0.7893548070313267, 0.7801724137931034, 0.7814475025484199, 0.7794561933534744, 0.788754859934204, 0.7866958773152759, 0.7926709778561631]

VERLAUF_4_TESTSET_ACCURACY = [0.6760958972815728, 0.7132109539572675, 0.7326712809710101, 0.7444076637576487, 0.7513291202728458, 0.7577490219681011, 0.7633664359514495, 0.7654729661952051, 0.7676798074029492, 0.774801885846123, 0.7761059283779718, 0.7793158792255993, 0.7799177450095296, 0.781422409469355, 0.785836091884843, 0.7871401344166917, 0.7867388905607383, 0.7829270739291805, 0.785535158992878, 0.7848329822449593]
VERLAUF_4_TESTSET_PRECISION = [0.6701611337604348, 0.6969753200964929, 0.7236918324000773, 0.733678955453149, 0.7485600794438928, 0.7484033288174956, 0.7534299516908213, 0.7594191522762951, 0.7645891226677253, 0.7595296794993363, 0.7675097276264592, 0.7884695147241808, 0.7926112510495382, 0.7715559000193761, 0.7977410583559925, 0.7918367346938775, 0.8096270598438855, 0.7768172888015717, 0.7839456869009584, 0.7766725180417398]
VERLAUF_4_TESTSET_RECALL = [0.6928944199116821, 0.7539140907266159, 0.7523083099156965, 0.7669610598153352, 0.7565234845443597, 0.7761942994781212, 0.7826174227217985, 0.776796467282216, 0.7731834604576475, 0.8038940184664793, 0.7918506623845845, 0.7631473303894019, 0.7579285427539141, 0.7992773986350863, 0.7655560016057809, 0.7788036932958651, 0.7494981934965878, 0.7936571657968687, 0.7880369329586512, 0.7992773986350863]
VERLAUF_4_TESTSET_F1 = [0.6813382019145366, 0.7243274515475846, 0.7377226650920186, 0.7499509322865555, 0.7525207147848657, 0.7620455217262785, 0.7677463818056512, 0.7680095256995435, 0.7688622754491018, 0.7810823988298391, 0.7794902193242441, 0.7756017951856384, 0.7748820028729735, 0.7851720398304249, 0.7813172180682167, 0.7852661404573973, 0.778403168647071, 0.7851469420174741, 0.785985985985986, 0.7878128400435256]

VERLAUF_5_TESTSET_ACCURACY = [0.6767980740294914, 0.7137125087772094, 0.7320694151870799, 0.7432039321897883, 0.7500250777409971, 0.7584511987160196, 0.7641689236633564, 0.7667770087270539, 0.7684822951148561, 0.7754037516300532, 0.7794161901895877, 0.7822248971812619, 0.7824255191092386, 0.7833283177851339, 0.7865382686327616, 0.7885444879125288, 0.788042933092587, 0.7860367138128197, 0.7873407563446685, 0.7871401344166917]
VERLAUF_5_TESTSET_PRECISION = [0.6702786377708978, 0.699475065616798, 0.7220834134153373, 0.7312834224598931, 0.7485029940119761, 0.7500971628449281, 0.7536148062463852, 0.7619798856241372, 0.7664408130729374, 0.7614871306005719, 0.7703516611618418, 0.7863108576084742, 0.7889870556811177, 0.7739805825242718, 0.7976637463496037, 0.7959225700164745, 0.8096265918411396, 0.7816887482697251, 0.7864291433146517, 0.7864583333333334]
VERLAUF_5_TESTSET_RECALL = [0.695303091128061, 0.748896025692493, 0.7541148133279807, 0.7685668406262545, 0.7527097551184263, 0.7747892412685669, 0.7846246487354476, 0.7755921316740265, 0.7719791248494581, 0.8016860698514653, 0.7958651144118828, 0.7747892412685669, 0.7707747892412685, 0.800080289040546, 0.76756322761943, 0.7757928542753915, 0.7529104777197912, 0.7934564431955038, 0.7886391007627459, 0.7880369329586512]
VERLAUF_5_TESTSET_F1 = [0.6825615763546798, 0.7233423807677395, 0.7377515954835543, 0.7494617341945586, 0.7506004803843074, 0.7622432859399684, 0.7688071590126856, 0.7687257535064159, 0.7692, 0.7810697174146867, 0.7829005824859315, 0.7805075320998888, 0.7797745964057264, 0.7868140544808527, 0.7823240589198036, 0.785728806668022, 0.7802392095683827, 0.7875286383105886, 0.7875325716576468, 0.7872468417886506]




if __name__ == "__main__":
    VERLAUF_1_TESTSET_MEAN_ACCURACY = np.mean(VERLAUF_1_TESTSET_ACCURACY)
    VERLAUF_2_TESTSET_MEAN_ACCURACY = np.mean(VERLAUF_2_TESTSET_ACCURACY)
    VERLAUF_3_TESTSET_MEAN_ACCURACY = np.mean(VERLAUF_3_TESTSET_ACCURACY)
    VERLAUF_4_TESTSET_MEAN_ACCURACY = np.mean(VERLAUF_4_TESTSET_ACCURACY)
    VERLAUF_5_TESTSET_MEAN_ACCURACY = np.mean(VERLAUF_5_TESTSET_ACCURACY)

    VERLAUF_1_TESTSET_MEAN_ACCURACY_SECOND_HALF = np.mean(VERLAUF_1_TESTSET_ACCURACY[-10:])
    VERLAUF_2_TESTSET_MEAN_ACCURACY_SECOND_HALF = np.mean(VERLAUF_2_TESTSET_ACCURACY[-10:])
    VERLAUF_3_TESTSET_MEAN_ACCURACY_SECOND_HALF = np.mean(VERLAUF_3_TESTSET_ACCURACY[-10:])
    VERLAUF_4_TESTSET_MEAN_ACCURACY_SECOND_HALF = np.mean(VERLAUF_4_TESTSET_ACCURACY[-10:])
    VERLAUF_5_TESTSET_MEAN_ACCURACY_SECOND_HALF = np.mean(VERLAUF_5_TESTSET_ACCURACY[-10:])

    VERLAUF_TESTSET_MEAN_ACCURACY =  np.mean([VERLAUF_1_TESTSET_MEAN_ACCURACY, VERLAUF_2_TESTSET_MEAN_ACCURACY, VERLAUF_3_TESTSET_MEAN_ACCURACY, VERLAUF_4_TESTSET_MEAN_ACCURACY, VERLAUF_5_TESTSET_MEAN_ACCURACY])
    VERLAUF_TESTSET_MEAN_ACCURACY_SECOND_HALF =  np.mean([VERLAUF_1_TESTSET_MEAN_ACCURACY_SECOND_HALF, VERLAUF_2_TESTSET_MEAN_ACCURACY_SECOND_HALF, VERLAUF_3_TESTSET_MEAN_ACCURACY_SECOND_HALF, VERLAUF_4_TESTSET_MEAN_ACCURACY_SECOND_HALF, VERLAUF_5_TESTSET_MEAN_ACCURACY_SECOND_HALF])
    print("\nVERLAUF_TESTSET_MEAN_ACCURACY: ", VERLAUF_TESTSET_MEAN_ACCURACY)
    print("VERLAUF_TESTSET_MEAN_ACCURACY_SECOND_HALF: ", VERLAUF_TESTSET_MEAN_ACCURACY_SECOND_HALF)



    VERLAUF_1_TESTSET_MEAN_PRECISION = np.mean(VERLAUF_1_TESTSET_PRECISION)
    VERLAUF_2_TESTSET_MEAN_PRECISION = np.mean(VERLAUF_2_TESTSET_PRECISION)
    VERLAUF_3_TESTSET_MEAN_PRECISION = np.mean(VERLAUF_3_TESTSET_PRECISION)
    VERLAUF_4_TESTSET_MEAN_PRECISION = np.mean(VERLAUF_4_TESTSET_PRECISION)
    VERLAUF_5_TESTSET_MEAN_PRECISION = np.mean(VERLAUF_5_TESTSET_PRECISION)

    VERLAUF_1_TESTSET_MEAN_PRECISION_SECOND_HALF = np.mean(VERLAUF_1_TESTSET_PRECISION[-10:])
    VERLAUF_2_TESTSET_MEAN_PRECISION_SECOND_HALF = np.mean(VERLAUF_2_TESTSET_PRECISION[-10:])
    VERLAUF_3_TESTSET_MEAN_PRECISION_SECOND_HALF = np.mean(VERLAUF_3_TESTSET_PRECISION[-10:])
    VERLAUF_4_TESTSET_MEAN_PRECISION_SECOND_HALF = np.mean(VERLAUF_4_TESTSET_PRECISION[-10:])
    VERLAUF_5_TESTSET_MEAN_PRECISION_SECOND_HALF = np.mean(VERLAUF_5_TESTSET_PRECISION[-10:])

    VERLAUF_TESTSET_MEAN_PRECISION =  np.mean([VERLAUF_1_TESTSET_MEAN_PRECISION, VERLAUF_2_TESTSET_MEAN_PRECISION, VERLAUF_3_TESTSET_MEAN_PRECISION, VERLAUF_4_TESTSET_MEAN_PRECISION, VERLAUF_5_TESTSET_MEAN_PRECISION])
    VERLAUF_TESTSET_MEAN_PRECISION_SECOND_HALF =  np.mean([VERLAUF_1_TESTSET_MEAN_PRECISION_SECOND_HALF, VERLAUF_2_TESTSET_MEAN_PRECISION_SECOND_HALF, VERLAUF_3_TESTSET_MEAN_PRECISION_SECOND_HALF, VERLAUF_4_TESTSET_MEAN_PRECISION_SECOND_HALF, VERLAUF_5_TESTSET_MEAN_PRECISION_SECOND_HALF])
    print("\nVERLAUF_TESTSET_MEAN_PRECISION: ", VERLAUF_TESTSET_MEAN_PRECISION)
    print("VERLAUF_TESTSET_MEAN_PRECISION_SECOND_HALF: ", VERLAUF_TESTSET_MEAN_PRECISION_SECOND_HALF)



    VERLAUF_1_TESTSET_MEAN_RECALL = np.mean(VERLAUF_1_TESTSET_RECALL)
    VERLAUF_2_TESTSET_MEAN_RECALL = np.mean(VERLAUF_2_TESTSET_RECALL)
    VERLAUF_3_TESTSET_MEAN_RECALL = np.mean(VERLAUF_3_TESTSET_RECALL)
    VERLAUF_4_TESTSET_MEAN_RECALL = np.mean(VERLAUF_4_TESTSET_RECALL)
    VERLAUF_5_TESTSET_MEAN_RECALL = np.mean(VERLAUF_5_TESTSET_RECALL)

    VERLAUF_1_TESTSET_MEAN_RECALL_SECOND_HALF = np.mean(VERLAUF_1_TESTSET_RECALL[-10:])
    VERLAUF_2_TESTSET_MEAN_RECALL_SECOND_HALF = np.mean(VERLAUF_2_TESTSET_RECALL[-10:])
    VERLAUF_3_TESTSET_MEAN_RECALL_SECOND_HALF = np.mean(VERLAUF_3_TESTSET_RECALL[-10:])
    VERLAUF_4_TESTSET_MEAN_RECALL_SECOND_HALF = np.mean(VERLAUF_4_TESTSET_RECALL[-10:])
    VERLAUF_5_TESTSET_MEAN_RECALL_SECOND_HALF = np.mean(VERLAUF_5_TESTSET_RECALL[-10:])

    VERLAUF_TESTSET_MEAN_RECALL=  np.mean([VERLAUF_1_TESTSET_MEAN_RECALL, VERLAUF_2_TESTSET_MEAN_RECALL, VERLAUF_3_TESTSET_MEAN_RECALL, VERLAUF_4_TESTSET_MEAN_RECALL, VERLAUF_5_TESTSET_MEAN_RECALL])
    VERLAUF_TESTSET_MEAN_RECALL_SECOND_HALF=  np.mean([VERLAUF_1_TESTSET_MEAN_RECALL_SECOND_HALF, VERLAUF_2_TESTSET_MEAN_RECALL_SECOND_HALF, VERLAUF_3_TESTSET_MEAN_RECALL_SECOND_HALF, VERLAUF_4_TESTSET_MEAN_RECALL_SECOND_HALF, VERLAUF_5_TESTSET_MEAN_RECALL_SECOND_HALF])
    print("\nVERLAUF_TESTSET_MEAN_RECALL: ", VERLAUF_TESTSET_MEAN_RECALL)
    print("VERLAUF_TESTSET_MEAN_RECALL_SECOND_HALF: ", VERLAUF_TESTSET_MEAN_RECALL_SECOND_HALF)


    VERLAUF_1_TESTSET_MEAN_F1 = np.mean(VERLAUF_1_TESTSET_F1)
    VERLAUF_2_TESTSET_MEAN_F1 = np.mean(VERLAUF_2_TESTSET_F1)
    VERLAUF_3_TESTSET_MEAN_F1 = np.mean(VERLAUF_3_TESTSET_F1)
    VERLAUF_4_TESTSET_MEAN_F1 = np.mean(VERLAUF_4_TESTSET_F1)
    VERLAUF_5_TESTSET_MEAN_F1 = np.mean(VERLAUF_5_TESTSET_F1)

    VERLAUF_1_TESTSET_MEAN_F1_SECOND_HALF = np.mean(VERLAUF_1_TESTSET_F1[-10:])
    VERLAUF_2_TESTSET_MEAN_F1_SECOND_HALF = np.mean(VERLAUF_2_TESTSET_F1[-10:])
    VERLAUF_3_TESTSET_MEAN_F1_SECOND_HALF = np.mean(VERLAUF_3_TESTSET_F1[-10:])
    VERLAUF_4_TESTSET_MEAN_F1_SECOND_HALF = np.mean(VERLAUF_4_TESTSET_F1[-10:])
    VERLAUF_5_TESTSET_MEAN_F1_SECOND_HALF = np.mean(VERLAUF_5_TESTSET_F1[-10:])

    VERLAUF_TESTSET_MEAN_F1 =  np.mean([VERLAUF_1_TESTSET_MEAN_F1, VERLAUF_2_TESTSET_MEAN_F1, VERLAUF_3_TESTSET_MEAN_F1, VERLAUF_4_TESTSET_MEAN_F1, VERLAUF_5_TESTSET_MEAN_F1])
    VERLAUF_TESTSET_MEAN_F1_SECOND_HALF =  np.mean([VERLAUF_1_TESTSET_MEAN_F1_SECOND_HALF, VERLAUF_2_TESTSET_MEAN_F1_SECOND_HALF, VERLAUF_3_TESTSET_MEAN_F1_SECOND_HALF, VERLAUF_4_TESTSET_MEAN_F1_SECOND_HALF, VERLAUF_5_TESTSET_MEAN_F1_SECOND_HALF])
    print("\nVERLAUF_TESTSET_MEAN_F1: ", VERLAUF_TESTSET_MEAN_F1)
    print("\VERLAUF_TESTSET_MEAN_F1_SECOND_HALF: ", VERLAUF_TESTSET_MEAN_F1_SECOND_HALF)