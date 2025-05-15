import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import subprocess
import code
from matplotlib.collections import PatchCollection

from collections import Counter

usePdfCrop = False

# color-blind friendly palette, from cold to warm-toned
colorsGlobal = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499', '#EE3377', '#CC3311', '#EE7733']

patternMisc = ''
# various gray tones
colorMisc =  ['#DDDDDD']
colorMiscAD =   ['#BBBBBB']
colorNoEval = ['#111111']


conferences = ["ECRTS", "RTAS", "RTSS"]
years = [2017,2018,2019,2020,2021,2022,2023,2024]
yearsTick = [str(item) for item in years]


df = pd.read_csv('data.csv', on_bad_lines='warn', sep=';')

typeCategory = {
    "benchmark": "Code",
    "taskset generator": "Tasks",
    "case study": "System",
    "microbenchmark": "Code",
    "graph generator": "Tasks",
    "industrial challenge": "System",
    "specification": "System",
    "dag generator": "Tasks",
    "software": "Unspecified",
    "model checker": "Unspecified",
    "axi message generator": "Unspecified",
    "testbed": "Unspecified",
    "network traffic generator": "Unspecified",
    "server workload generator": "Unspecified",
    "workload generator": "Unspecified",
    "dataset": "Unspecified",
    "memory traffic generator": "Unspecified",
    "communication traffic generator": "Unspecified",
    "benchmark generator": "Code",
    "can messages": "Unspecified",
    "can message generator": "Unspecified",
    "os": "Unspecified",
    "network topology generator": "Unspecified",
    "tool": "Unspecified",
    "ressource access graph generator": "Unspecified",
    "whole system generator": "System",
    "input variability curve generator": "Unspecified",
    "event generator": "Unspecified",
    "execution time distribution generator": "Unspecified",
    "ros2 chain generator": "Unspecified",
    "data stream generator": "Unspecified",
    "misc generator": "Unspecified",
    "ros message generator": "Unspecified",
    "gpu workload generator": "Unspecified",
    "data transaction generator": "Unspecified",
    "pwcet generator": "Unspecified",
    "can traffic generator":  "Unspecified",
    "dma traffic generator": "Unspecified",
    "schedulability analysis tool" : "Tasks",
}

def cleanData():
    cleanDF = df
    cleanDF = cleanDF.replace(np.nan, '')
    cleanDF.columns = map(lambda x: str(x).upper(), df.columns)
    cleanDF = cleanDF.replace(r"^ +| +$", r"", regex=True)
    for column in ['AUTHOR1','AUTHOR2','AUTHOR3','AUTHOR4','AUTHOR5','AUTHOR6','AUTHOR7','AUTHOR8','AUTHOR9','AUTHOR10','AUTHOR11','AUTHOR12','AUTHOR13','AUTHOR14','AUTHOR15']:
            cleanDF[column] = cleanDF[column].apply(lambda x: x.upper())
    for column in ['SPECIFIER1','SPECIFIER2','SPECIFIER3','SPECIFIER4','SPECIFIER5','SPECIFIER6','SPECIFIER7']:
            cleanDF[column] = cleanDF[column].apply(lambda x: x.upper())
    for column in ['QUALIFIER1','QUALIFIER2','QUALIFIER3','QUALIFIER4','QUALIFIER5','QUALIFIER6','QUALIFIER7']:
            cleanDF[column] = cleanDF[column].apply(lambda x: x.upper())
    for column in ['FIELD1','FIELD2','FIELD3','FIELD4','FIELD5','FIELD6','FIELD7']:
            cleanDF[column] = cleanDF[column].apply(lambda x: x.upper())
    return cleanDF


def correctData():

    df['HASEVAL'] = False

    df.loc[ (df['TYPE1'] != "") & (df['QUALIFIER1'] == ""), 'QUALIFIER1'] = 'UNSPECIFIED'
    df.loc[ (df['TYPE2'] != "") & (df['QUALIFIER2'] == ""), 'QUALIFIER2'] = 'UNSPECIFIED'
    df.loc[ (df['TYPE3'] != "") & (df['QUALIFIER3'] == ""), 'QUALIFIER3'] = 'UNSPECIFIED'
    df.loc[ (df['TYPE4'] != "") & (df['QUALIFIER4'] == ""), 'QUALIFIER4'] = 'UNSPECIFIED'
    df.loc[ (df['TYPE5'] != "") & (df['QUALIFIER5'] == ""), 'QUALIFIER5'] = 'UNSPECIFIED'
    df.loc[ (df['TYPE6'] != "") & (df['QUALIFIER6'] == ""), 'QUALIFIER6'] = 'UNSPECIFIED'
    df.loc[ (df['TYPE7'] != "") & (df['QUALIFIER7'] == ""), 'QUALIFIER7'] = 'UNSPECIFIED'


    # if we want to include GENERIC as a kind on its own
    # df.loc[ (df['TYPE1'] != "") & (df['FIELD1'] == ""), 'FIELD1'] = 'GENERIC'
    # df.loc[ (df['TYPE2'] != "") & (df['FIELD2'] == ""), 'FIELD2'] = 'GENERIC'
    # df.loc[ (df['TYPE3'] != "") & (df['FIELD3'] == ""), 'FIELD3'] = 'GENERIC'
    # df.loc[ (df['TYPE4'] != "") & (df['FIELD4'] == ""), 'FIELD4'] = 'GENERIC'
    # df.loc[ (df['TYPE5'] != "") & (df['FIELD5'] == ""), 'FIELD5'] = 'GENERIC'
    # df.loc[ (df['TYPE6'] != "") & (df['FIELD6'] == ""), 'FIELD6'] = 'GENERIC'
    # df.loc[ (df['TYPE7'] != "") & (df['FIELD7'] == ""), 'FIELD7'] = 'GENERIC'

    df.loc[ (df['TYPE1'] != "case study") & (df['FIELD1'] != ""), 'FIELD1'] = ''
    df.loc[ (df['TYPE2'] != "case study") & (df['FIELD2'] != ""), 'FIELD2'] = ''
    df.loc[ (df['TYPE3'] != "case study") & (df['FIELD3'] != ""), 'FIELD3'] = ''
    df.loc[ (df['TYPE4'] != "case study") & (df['FIELD4'] != ""), 'FIELD4'] = ''
    df.loc[ (df['TYPE5'] != "case study") & (df['FIELD5'] != ""), 'FIELD5'] = ''
    df.loc[ (df['TYPE6'] != "case study") & (df['FIELD6'] != ""), 'FIELD6'] = ''
    df.loc[ (df['TYPE7'] != "case study") & (df['FIELD7'] != ""), 'FIELD7'] = ''

    df.loc[ (df['TYPE1'] == "case study") & (df['FIELD1'] == ""), 'FIELD1'] = 'MISC'
    df.loc[ (df['TYPE2'] == "case study") & (df['FIELD2'] == ""), 'FIELD2'] = 'MISC'
    df.loc[ (df['TYPE3'] == "case study") & (df['FIELD3'] == ""), 'FIELD3'] = 'MISC'
    df.loc[ (df['TYPE4'] == "case study") & (df['FIELD4'] == ""), 'FIELD4'] = 'MISC'
    df.loc[ (df['TYPE5'] == "case study") & (df['FIELD5'] == ""), 'FIELD5'] = 'MISC'
    df.loc[ (df['TYPE6'] == "case study") & (df['FIELD6'] == ""), 'FIELD6'] = 'MISC'
    df.loc[ (df['TYPE7'] == "case study") & (df['FIELD7'] == ""), 'FIELD7'] = 'MISC'

    # Set HASEVAL
    df.loc[df['TYPE1'] != "",'HASEVAL'] = True
    df.loc[df['TYPE1'] == "",'HASEVAL'] = False
    # computeNrofEval
    df.loc[df['TYPE1'] == "",'NROFEVAL'] = 0
    df.loc[df['TYPE1'] != "",'NROFEVAL'] = 1
    df.loc[df['TYPE2'] != "",'NROFEVAL'] = 2
    df.loc[df['TYPE3'] != "",'NROFEVAL'] = 3
    df.loc[df['TYPE4'] != "",'NROFEVAL'] = 4
    df.loc[df['TYPE5'] != "",'NROFEVAL'] = 5
    df.loc[df['TYPE6'] != "",'NROFEVAL'] = 6
    df.loc[df['TYPE7'] != "",'NROFEVAL'] = 7


def arrayCombine(x, y):
    result = []
    for i in x:
        result.append(i)
    for i in y:
        result.append(i)
    return result


def allTypes(df):
    types = arrayCombine(df.TYPE1.array, df.TYPE2.array)
    types = arrayCombine(types, df.TYPE3.array)
    types = arrayCombine(types, df.TYPE4.array)
    types = arrayCombine(types, df.TYPE5.array)
    types = arrayCombine(types, df.TYPE6.array)
    types = arrayCombine(types, df.TYPE7.array)
    types = [str(item) for item in types]
    types = [item for item in types if item != ""]
    number = len(types)
    typeMap = Counter(types)
    return typeMap, number


def allCategories(df):
    types = arrayCombine(df.TYPE1.array, df.TYPE2.array)
    types = arrayCombine(types, df.TYPE3.array)
    types = arrayCombine(types, df.TYPE4.array)
    types = arrayCombine(types, df.TYPE5.array)
    types = arrayCombine(types, df.TYPE6.array)
    types = arrayCombine(types, df.TYPE7.array)
    types = [str(item) for item in types]
    types = [item for item in types if item != ""]
    categories = [typeCategory[item] for item in types if item != ""]
    number = len(categories)
    categoryMap  = Counter(categories)
    return categoryMap, number


def allSpecifier(df):
    specifier = arrayCombine(df.SPECIFIER1.array, df.SPECIFIER2.array)
    specifier = arrayCombine(specifier, df.SPECIFIER3.array)
    specifier = arrayCombine(specifier, df.SPECIFIER4.array)
    specifier = arrayCombine(specifier, df.SPECIFIER5.array)
    specifier = arrayCombine(specifier, df.SPECIFIER6.array)
    specifier = arrayCombine(specifier, df.SPECIFIER7.array)
    specifier = [str(item) for item in specifier]
    specifier = [item for item in specifier if item != ""]
    number = len(specifier)
    specifierMap = Counter(specifier)
    return specifierMap, number


def allQualifier(df):
    qualifier = arrayCombine(df.QUALIFIER1.array, df.QUALIFIER2.array)
    qualifier = arrayCombine(qualifier, df.QUALIFIER3.array)
    qualifier = arrayCombine(qualifier, df.QUALIFIER4.array)
    qualifier = arrayCombine(qualifier, df.QUALIFIER5.array)
    qualifier = arrayCombine(qualifier, df.QUALIFIER6.array)
    qualifier = arrayCombine(qualifier, df.QUALIFIER7.array)
    qualifier = [str(item) for item in qualifier]
    qualifier = [item for item in qualifier if item != ""]
    number = len(qualifier)
    qualifierMap = Counter(qualifier)
    return qualifierMap, number


def allFields(df):
    field = arrayCombine(df.FIELD1.array, df.FIELD2.array)
    field = arrayCombine(field, df.FIELD3.array)
    field = arrayCombine(field, df.FIELD4.array)
    field = arrayCombine(field, df.FIELD5.array)
    field = arrayCombine(field, df.FIELD6.array)
    field = arrayCombine(field, df.FIELD7.array)
    field = [str(item) for item in field]
    field = [item for item in field if item != ""]
    number = len(field)
    fieldMap = Counter(field)
    return fieldMap, number


def allAuthors(df):
    authors = arrayCombine(df.AUTHOR1.array, df.AUTHOR2.array)
    authors = arrayCombine(authors, df.AUTHOR3.array)
    authors = arrayCombine(authors, df.AUTHOR4.array)
    authors = arrayCombine(authors, df.AUTHOR5.array)
    authors = arrayCombine(authors, df.AUTHOR6.array)
    authors = arrayCombine(authors, df.AUTHOR7.array)
    authors = arrayCombine(authors, df.AUTHOR8.array)
    authors = arrayCombine(authors, df.AUTHOR9.array)
    authors = arrayCombine(authors, df.AUTHOR10.array)
    authors = arrayCombine(authors, df.AUTHOR11.array)
    authors = arrayCombine(authors, df.AUTHOR12.array)
    authors = arrayCombine(authors, df.AUTHOR13.array)
    authors = arrayCombine(authors, df.AUTHOR14.array)
    authors = arrayCombine(authors, df.AUTHOR15.array)
    authors = [str(item) for item in authors]
    authors = [item for item in authors if item != ""]
    number = len(authors)
    authorMap = Counter(authors)
    return authorMap, number


df = cleanData()
correctData()


dfComplete = df

authorMap, number = allAuthors(df)
typeMap, number = allTypes(df)
categoryMap, number = allCategories(df)
specifierMap, number = allSpecifier(df)
qualifierMap, number = allQualifier(df)
fieldMap, number = allFields(df)


def computeQueryPerYear(identifier,allX,yearsOfInterest=years):
    xPerYear = []
    for year in yearsOfInterest:
        num = 0
        for conf in conferences:
            queryString = 'CONFERENCE == "' + conf + '" and YEAR == ' + str(year)
            # print(queryString)

            dfResult = df.query(queryString)
            count, discardNum = allX(dfResult)
            # count = dfResult.count()
            num += count[identifier]

            # dfResult = dfResult.query(columnQuery)
            # num += len(dfResult)
        xPerYear.append(num)
    return xPerYear


# def computeArtifactsPerConference():
#     artifactsPerConference = []
#     # Number of Artifacts per year and conference
#     for conf in conferences:
#         artifactsPerConferenceConf = []
#         for year in years:
#
#             if (year == 2024 and conf == "RTSS"):
#                 artifactsPerConferenceConf.append(np.nan)
#                 continue
#
#             queryString = 'CONFERENCE == "' + conf + '" and YEAR == ' + str(year)
#
#             dfResult = df.query(queryString)
#             NumPapers  = dfResult.count().iloc[0]
#
#             dfResult = dfResult.query('ARTIFACTEVALUATION')
#             NumArtifacts = dfResult.count().iloc[0]
#
#             dfResult = dfResult.query('ARTIFACTWASAVAILABLE')
#             NumArtifactsWasAvailable = dfResult.count().iloc[0]
#
#             dfResult = dfResult.query('ARTIFACTISAVAILABLE')
#             NumArtifactsIsAvailable = dfResult.count().iloc[0]
#
#             artifactsPerConferenceConf.append(NumArtifacts/NumPapers)
#
#             print(conf + " " + str(year) + " " + str(NumPapers) + " " + str(NumArtifacts) + " " + str(NumArtifactsWasAvailable) + " " + str(NumArtifactsIsAvailable))
#             print(conf + " " + str(year) + " " + str(NumPapers) + " " + str(NumArtifacts/NumPapers) + " " + str(NumArtifactsWasAvailable/NumPapers) + " " + str(NumArtifactsIsAvailable/NumPapers))
#             # print(dfResult)
#         artifactsPerConference.append(artifactsPerConferenceConf)
#     return artifactsPerConference


def filterToDetail(amount, dataMap, withMISC):
    labels = []
    data = []
    miscAmount = 0;
    for key, value in dataMap.items():
        if (value >= amount and key != "MISC"):
            labels.append(key)
            data.append(value)
        else:
            miscAmount +=value

    if (withMISC):
        labels.append("Misc")
        data.append(miscAmount)

    # print(len(labels))

    return labels,data


def filterTo(amount, dataMap):
    return filterToDetail(amount, dataMap, True)


def filterToDetailSort(amount, dataMap, withMISC):
    dataIn = dataMap.most_common()
    dataIn.reverse()
    labels = []
    data = []
    miscAmount = 0;
    while (len(dataIn)>0):
        key, value = dataIn.pop()
        if (value >= amount and key != "MISC"):
            labels.append(key)
            data.append(value)
        else:
            miscAmount +=value


    if (withMISC):
        labels.append("Misc")
        data.append(miscAmount)


    # print(len(labels))

    return labels,data


def filterToSort(amount, dataMap):
    return filterToDetailSort(amount, dataMap, True)


# def plotXPerConferencePrint(xPerConference,name):
#     # artifactsPerConference = computeArtifactsPerConference()
#     fig, ax = plt.subplots()
#     ax.plot(yearsTick, xPerConference[0], label="ECRTS")
#     ax.plot(yearsTick, xPerConference[1], label="RTAS")
#     ax.plot(yearsTick, xPerConference[2], label="RTSS")
#     ax.legend()
#     if (name != ""):
#         plt.savefig(name, format="pdf", bbox_inches="tight")
#     else:
#         plt.show()
#
# def plotXPerConference(xPerConference):
#     plotXPerConferencePrint(xPerConference,"")


def computeXPerConference(allX):
    xPerConference = []
    for conf in conferences:
        xPerConferenceConf = []
        for year in years:
            queryString = 'CONFERENCE == "' + conf + '" and YEAR == ' + str(year)
            # print(queryString)

            dfResult = df.query(queryString)
            typeMap, num = allX(dfResult)
            x = dfResult.count().iloc[0]
            # print(typeMap)

            xPerConferenceConf.append(typeMap)

        xPerConference.append(xPerConferenceConf)
    return xPerConference


def extractKeys(counter):
    keys = []
    for key, value in counter.items():
        keys.append(key)
    return keys


def processXPerConference(data,keys,normalize,misc):
    # print(keys)
    # xPerConference = []
    data.reverse()

    dataFrames = {}

    keysAdded = ["Year"] + keys


    for conf in conferences:
        confData = data.pop()
        confData.reverse()

        plotDataC = []
        for year in years:
            yearConfData = confData.pop()

            plotDataYC = []
            notMISC = 0
            for key in keys:
                # print("key= " + str(key))
                YCF = yearConfData[key]
                # print("YCF= " + str(YCF))
                notMISC += YCF
                plotDataYC.append(YCF)
                # print(key)
                # print(plotDataYC)

            # plotDataYC.reverse()

            num = sum(yearConfData.values())
            # print(notMISC)

            if(misc):
                plotDataYC[keys.index("Misc")] = (num - notMISC) - plotDataYC[keys.index("Misc")]
                # if plotDataYC["MISC"] != 0:
                # print("Warnung: Misc " + str(keys.index("MISC")) )

            if(normalize):
                # print("\n")
                # print(0)

                plotDataYC = list(map(lambda x: x/num,plotDataYC))
                # print(plotDataYC)
                # print(num)

            plotDataYC = [year] + plotDataYC
            # print(keys)
            # print(plotDataYC)

            plotDataC.append(plotDataYC)

        # print(plotDataC)

        dfPlot = pd.DataFrame(plotDataC,columns=keysAdded)
        # print(dfPlot)
        dataFrames[conf] = dfPlot
        # print(dataFrames)

    return dataFrames

### Plotting Helpers

def plotPiChartPrint(labels,data,name):
    fig, ax = plt.subplots()
    wedges,_ =  ax.pie(data, labels=labels)
    if (name != ""):
        plt.savefig(name, format="pdf", bbox_inches="tight")
    else:
        plt.show()


def plotPiChartPattern(labels,data):
    plotPiChartPrint(labels,data,"")


def plotPiChartPrintPattern(labels,data,name,pattern,color):
    patterns = itertools.cycle(pattern)
    colors = itertools.cycle(color)
    fig, ax = plt.subplots()
    wedges,_ =  ax.pie(data, labels=labels)
    for pie_wedge in wedges:
        pie_wedge.set_hatch(next(patterns))
        pie_wedge.set_facecolor(next(colors))
        pie_wedge.set_alpha(0.99)
    if (name != ""):
        plt.savefig(name, format="pdf", bbox_inches="tight")
    else:
        plt.show()


def plotPiChartPattern(labels,data,pattern):
    plotPiChartPrint(labels,data,"",pattern)


def plotStackedBarChartPrint(data,name):
    ax = data.plot(x='Year', kind='bar', stacked=True)
    if (name != ""):
        plt.savefig(name, format="pdf", bbox_inches="tight")
    else:
        plt.show()


def plotStackedBarChart(plot):
    plotStackedBarChartPrint(plot,"")



### Generate Graphs

def reproduciblePerYearConfCompute(normalize,strict):

    artString = "ARTIFACTEVALUATION"
    if (strict):
        artString = "ARTIFACTISAVAILABLE"

    queryString = ('HASEVAL')
    dfEval = df.query(queryString)

    # dfEval = df

    # print("reproduciblePerYearConfCompute: Nr of papers" + str(len(dfEval)))

    paperSum = 0

    xPerConference = []
    for conf in conferences:
        xPerConferenceConf = []
        for year in years:
            xPerConferenceConfYear = []
            # print(conf + " " + str(year))

            queryString = 'CONFERENCE == "' + conf + '" and YEAR == ' + str(year)
            dfResult = dfEval.query(queryString)
            paperSum += len(dfResult)
            # print(conf + " " + str(year) + " Size: " +  str(len(dfResult)) + " " + str(paperSum))

            # PROPRIETARY?
            queryString = 'PROPRIETARY1 != "yes" and PROPRIETARY2 != "yes" and PROPRIETARY3 != "yes" and PROPRIETARY4 != "yes" and PROPRIETARY5 != "yes" and PROPRIETARY6 != "yes" and PROPRIETARY7 != "yes"'
            dataSetSize = len(dfResult)
            dfResult = dfResult.query(queryString)
            newDataSetSize = len(dfResult)
            removedItems = dataSetSize-newDataSetSize
            xPerConferenceConfYear.append(removedItems)

            dataSetSize = len(dfResult)
            for string in ["CUSTOM", "UNSPECIFIED"]:

                # QUALIFIER EXTENDED
                queryString = 'QUALIFIER1 != "' + string + '" and QUALIFIER2 != "' + string + '" and QUALIFIER3 != "' + string + '" and QUALIFIER4 != "' + string + '" and QUALIFIER5 != "' + string + '" and QUALIFIER6 != "' + string + '" and QUALIFIER7 != "' + string + '" or ' +  artString
                dfResult = dfResult.query(queryString)
            newDataSetSize = len(dfResult)
            removedItems = dataSetSize-newDataSetSize
            xPerConferenceConfYear.append(removedItems)

            for string in ["EXTENDED",  "REFERENCED", "NAMED"]:

                # QUALIFIER EXTENDED
                queryString = 'QUALIFIER1 != "' + string + '" and QUALIFIER2 != "' + string + '" and QUALIFIER3 != "' + string + '" and QUALIFIER4 != "' + string + '" and QUALIFIER5 != "' + string + '" and QUALIFIER6 != "' + string + '" and QUALIFIER7 != "' + string + '" or ' +  artString
                dataSetSize = len(dfResult)
                dfResult = dfResult.query(queryString)
                newDataSetSize = len(dfResult)
                removedItems = dataSetSize-newDataSetSize
                xPerConferenceConfYear.append(removedItems)


            queryString = 'not ' +  artString
            dataSetSize = len(dfResult)
            dfResult = dfResult.query(queryString)
            newDataSetSize = len(dfResult)
            removedItems = dataSetSize-newDataSetSize

            # print(dfResult)

            xPerConferenceConfYear.append(removedItems)

            # print(xPerConferenceConfYear)
            # print(conf + " " + str(year) +"\n")


            if (normalize):
                total = sum(xPerConferenceConfYear)
                xPerConferenceConfYear = list(map(lambda x: x/total,xPerConferenceConfYear))
                # print(xPerConferenceConfYear)
                # print(conf + " " + str(year) +"\n")

            xPerConferenceConf.append(xPerConferenceConfYear)

        xPerConference.append(xPerConferenceConf)
    return xPerConference


def processReproduciblePerYearConf(data,keys):

    data.reverse()

    dataFrames = {}
    totalRep = {}
    totalRepList = [0,0,0,0,0,0]
    keysAdded = ["Year"] + keys


    for conf in conferences:
        confData = data.pop()
        confData.reverse()
    #
        plotDataC = []
        for year in years:
            plotDataYC = confData.pop()
            totalRepList = np.add(totalRepList,plotDataYC)

            plotDataYC = [year] + plotDataYC

            plotDataC.append(plotDataYC)

        dfPlot = pd.DataFrame(plotDataC,columns=keysAdded)

        dataFrames[conf] = dfPlot

    return dataFrames, totalRepList


def export_legend(ax, filename="legend.pdf"):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False)
    fig  = legend.figure
    fig.canvas.draw()
    # bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, format="pdf", bbox_inches="tight")


def augmentKeys(keys, data):
    newkeys = []
    i = 0
    for key in keys:
        if (data[i] != 0):
            newkeys.append(keys[i] + " ("+str(data[i]) + ")")
        else:
            newkeys.append("")
        i += 1

    return newkeys

def defineAuthorDF(thresholdAuthors):

    keysAuthor, dataAuthors = filterToDetailSort(thresholdAuthors,authorMap,False)

    dfOthers = df
    dfAuthors = pd.DataFrame()

    for author in keysAuthor:

        queryString = 'AUTHOR1 == "' + author + '" or AUTHOR2 == "' + author + '" or AUTHOR3 == "' + author + '" or AUTHOR4 == "' + author + '" or AUTHOR5 == "' + author \
            + '" or AUTHOR6 == "' + author + '" or AUTHOR7 == "' + author + '" or AUTHOR8 == "' + author + '" or AUTHOR9 == "' + author + '" or AUTHOR10 == "' + author \
            + '" or AUTHOR11 == "' + author + '" or AUTHOR12 == "' + author + '" or AUTHOR13 == "' + author + '" or AUTHOR14 == "' + author + '" or AUTHOR15 == "' + author + '"'

        notQueryString = 'AUTHOR1 != "' + author + '" and AUTHOR2 != "' + author + '" and AUTHOR3 != "' + author + '" and AUTHOR4 != "' + author + '" and AUTHOR5 != "' + author \
            + '" and AUTHOR6 != "' + author + '" and AUTHOR7 != "' + author + '" and AUTHOR8 != "' + author + '" and AUTHOR9 != "' + author + '" and AUTHOR10 != "' + author \
            + '" and AUTHOR11 != "' + author + '" and AUTHOR12 != "' + author + '" and AUTHOR13 != "' + author + '" and AUTHOR14 != "' + author + '" and AUTHOR15 != "' + author + '"'

        queryString2 = 'TYPE1 != ""'

        dfAuthor = df.query(queryString) # only from Authors author
        dfAuthors = pd.concat((dfAuthors,dfAuthor))

        dfOthers = dfOthers.query(notQueryString)

    dfAuthors = dfAuthors.drop_duplicates()
    return dfAuthors, dfOthers

def authorUsesX(thresholdAuthors, allX, normalize, anonymized):
    resultData = []

    keysAuthor, dataAuthors = filterToDetailSort(thresholdAuthors,authorMap,False)

    for author in keysAuthor:

        queryString = 'AUTHOR1 == "' + author + '" or AUTHOR2 == "' + author + '" or AUTHOR3 == "' + author + '" or AUTHOR4 == "' + author + '" or AUTHOR5 == "' + author \
            + '" or AUTHOR6 == "' + author + '" or AUTHOR7 == "' + author + '" or AUTHOR8 == "' + author + '" or AUTHOR9 == "' + author + '" or AUTHOR10 == "' + author \
            + '" or AUTHOR11 == "' + author + '" or AUTHOR12 == "' + author + '" or AUTHOR13 == "' + author + '" or AUTHOR14 == "' + author + '" or AUTHOR15 == "' + author + '"'


        dfAuthors = df.query(queryString) # only from Authors author


        mapping, num = allX(dfAuthors)

        # print(str(len(dfAuthors)))
        # print(mapping)
        # print(num)

        # if normalize:
        #     num = num/len(dfAuthors)
        #     print(num)

        resultData.append(num)

    i = 0;
    keysAnonym = []
    for author in keysAuthor:
        i += 1
        keysAnonym.append("Author " + str(i))

    if (anonymized):
        keysAuthor = keysAnonym

    return keysAuthor, resultData

def authorInformation(noEvalOption,thresholdAuthors,thresholdX,allX,xMap,XMisc,normalize,anonymized, noX, otherPapers=False):
    resultData = []

    keysAuthor, dataAuthors = filterToDetailSort(thresholdAuthors,authorMap,False)
    keysX, dataX = filterToDetailSort(thresholdX,xMap,XMisc)

    miscDf = df
    for author in keysAuthor:

        queryString = 'AUTHOR1 == "' + author + '" or AUTHOR2 == "' + author + '" or AUTHOR3 == "' + author + '" or AUTHOR4 == "' + author + '" or AUTHOR5 == "' + author \
            + '" or AUTHOR6 == "' + author + '" or AUTHOR7 == "' + author + '" or AUTHOR8 == "' + author + '" or AUTHOR9 == "' + author + '" or AUTHOR10 == "' + author \
            + '" or AUTHOR11 == "' + author + '" or AUTHOR12 == "' + author + '" or AUTHOR13 == "' + author + '" or AUTHOR14 == "' + author + '" or AUTHOR15 == "' + author + '"'

        notQueryString = 'AUTHOR1 != "' + author + '" and AUTHOR2 != "' + author + '" and AUTHOR3 != "' + author + '" and AUTHOR4 != "' + author + '" and AUTHOR5 != "' + author \
            + '" and AUTHOR6 != "' + author + '" and AUTHOR7 != "' + author + '" and AUTHOR8 != "' + author + '" and AUTHOR9 != "' + author + '" and AUTHOR10 != "' + author \
            + '" and AUTHOR11 != "' + author + '" and AUTHOR12 != "' + author + '" and AUTHOR13 != "' + author + '" and AUTHOR14 != "' + author + '" and AUTHOR15 != "' + author + '"'

        queryString2 = 'TYPE1 != ""'

        dfAuthor = df.query(queryString) # only from Authors author
        miscDf = miscDf.query(notQueryString)

        paperByAuthor = len(dfAuthor)
        dfAuthorEval = dfAuthor.query(queryString2) # only from Authors author
        noEval = paperByAuthor - len(dfAuthorEval)
        allXbyAuthor, num = allX(dfAuthorEval)
        MISCAmount = 0

        dataByAuthor = []
        for key in keysX:
            if (key != "MISC"):
                if (key != "Misc"):
                    amount = allXbyAuthor[key]
                    dataByAuthor.append(amount)
                    num -= amount
                else:
                    amount = allXbyAuthor[key]
                    dataByAuthor.append(amount + num + MISCAmount)
            else:
                MISCAmount = allXbyAuthor[key]

        if (np.sum(dataByAuthor) == 0):
            noEval = 1

        dataByAuthor.append(noEval)

        total = sum(dataByAuthor)
        if (normalize and total):
            dataByAuthor = list(map(lambda x: x/total,dataByAuthor))

        # dataByAuthor.reverse()
        resultData.append(dataByAuthor)

    paperByAuthor = len(miscDf)
    dfAuthorEval = miscDf.query(queryString2) # only from Authors author
    noEval = paperByAuthor - len(dfAuthorEval)
    allXbyAuthor, num = allX(dfAuthorEval)
    MISCAmount = 0

    if (otherPapers):
        dataByAuthor = []
        for key in keysX:
            if (key != "MISC"):
                if (key != "Misc"):
                    amount = allXbyAuthor[key]
                    dataByAuthor.append(amount)
                    num -= amount
                else:
                    amount = allXbyAuthor["key"]
                    dataByAuthor.append(amount + num + MISCAmount)
            else:
                MISCAmount = allXbyAuthor[key]

        if (np.sum(dataByAuthor) == 0):
            # print(dfAuthorEval)
            noEval = 1

        dataByAuthor.append(noEval)

        total = sum(dataByAuthor)
        if (normalize and total):
            dataByAuthor = list(map(lambda x: x/total,dataByAuthor))

        resultData.append(dataByAuthor)

    if ("MISC" in keysX):
        keysX.remove("MISC")

    # keysX.reverse()

    if (noEvalOption):
        keysX.append(noX)
    else:
        keysX.append("Not specified")
    # keysX.reverse()

    i = 0;
    keysAnonym = []
    for author in keysAuthor:
        i += 1
        keysAnonym.append("Author " + str(i))

    if(otherPapers):
        keysAuthor.append("Other paper")
        keysAnonym.append("Other paper")

    if (anonymized):
        keysAuthor = keysAnonym

    return keysAuthor, keysX, resultData



def plotAuthorInformation (keysAuthor, keysX, resultData, name, pattern, colors, size1, size2):
    plt.rcParams["figure.figsize"] = (size1,size2)

    N = len(keysX)
    M = len(keysAuthor)
    ylabels = keysX
    xlabels = keysAuthor
    x, y = np.meshgrid(np.arange(M), np.arange(N))
    s = np.array(resultData, dtype='float32')
    s = s.transpose()
    c = np.random.rand(1, 1)

    fig, ax = plt.subplots()
    plt.xticks(rotation=90)
    # R = s
    R = s/s.max()/2
    circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
    col = PatchCollection(circles)
    ax.add_collection(col)

    col.set_facecolor(colorExtender(colors,len(keysAuthor)))

    ax.set(xticks=np.arange(M), yticks=np.arange(N), xticklabels=xlabels, yticklabels=ylabels)
    ax.set_xticks(np.arange(M+1)-0.5, minor=True)
    ax.set_yticks(np.arange(N+1)-0.5, minor=True)
    ax.grid(which='minor')

    plt.savefig(name, format="pdf", bbox_inches="tight")


def plotDatePerYear(data,pattern,color,conf,name,size1,size2,legend,exportLegend):
    patterns = itertools.cycle(pattern)
    colors = itertools.cycle(color)
    ax = data.plot(x='Year',xlabel=conf, kind='bar', stacked=True, cmap='gist_ncar',alpha=0.99,figsize=(size1,size2), width=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    for container in ax.containers:
        hatch = next(patterns)
        color = next(colors)
        for artist in container:
            artist.set_facecolor(color)
            # artist.set_alpha(0.1)
            artist.set_hatch(hatch)
    if(not legend):
        ax.get_legend().remove()
    if(exportLegend):
        export_legend(ax,"leg_"+ name)
    else:
        plt.savefig(name, format="pdf", bbox_inches="tight")


def unifyKeys(keysG,keys,values):

    valuesNew = []
    valuesDict = dict(zip(keys, values))
    miscAmount = sum(values)

    for key in keysG:
        if (key in keys):
            valuesNew.append(valuesDict[key])
            miscAmount -= valuesDict[key]
        else:
            valuesNew.append(0)

    valuesNew[len(valuesNew)-1] = miscAmount

    return valuesNew


def reproducible(normalize=True,strict=True):
    size1 = 3
    size2 = 4
    patterns = ['xxx',          '**',         '...',         'OO',      'O',          '++',         '']#('/', '\\', '+', 'o', '*', '**', '***', '.')
    colors = colorsGlobal#['#D55E00','#E69F00','#F0E442','#56B4E9','#0072B2','#009E73']
    keys = ["Restricted", "Custom/Unspecified", "Referenced","Extended", "Named", "Artifact"]
    data = reproduciblePerYearConfCompute(False,strict)
    dataFramesPlots, totalRep = processReproduciblePerYearConf(data,keys)
    # print("Number of fields: " +  str(np.sum(totalRep)))
    newKeys = augmentKeys(keys, totalRep)
    plt.rcParams["figure.figsize"] = (5,3)
    plotPiChartPrintPattern(newKeys, totalRep, "reprodDistribution.pdf",patterns,colors)
    data = reproduciblePerYearConfCompute(normalize,strict)
    dataFramesPlots, totalRep = processReproduciblePerYearConf(data,keys)
    for conf in conferences:
        plotDatePerYear(dataFramesPlots[conf],patterns,colors,conf,filePrefix+"reprodDistribution"+conf+".pdf",size1,size2,False,True)
        plotDatePerYear(dataFramesPlots[conf],patterns,colors,conf,filePrefix+"reprodDistribution"+conf+".pdf",size1,size2,False,False)
        if(usePdfCrop):
            subprocess.call(["pdfcrop", filePrefix+"leg_reprodDistribution"+conf+".pdf"])
        # plotStackedBarChartPrint(dataFramesPlots[conf], "reprodDistribution"+conf+".pdf")

def benchmarks(threshold=12,normalize=True,misc=True):
    size1 = 3
    size2 = 4

    colorsBench = colorsGlobal
    
    if (filePrefix != ""):
        specifierMapL, number = allSpecifier(df)
        thresholdL = 0
    else:
        specifierMapL = specifierMap
        thresholdL = threshold

    keys, values = filterToDetailSort(thresholdL,specifierMapL,(misc & (filePrefix == "")))


    if (filePrefix != ""):
            keysG, _ = filterToDetailSort(threshold,specifierMapGlobal,True)
            values = unifyKeys(keysG,keys,values)
            keys = keysG
            keys.pop()
            values.pop()



    patterns = ('--', '///', '++', '..', 'oo', '---', '\\\\\\', '||','**', 'OO', 'ooo', '...', patternMisc, '...', 'OOO', '***')
    colors = colorsBench[0:(len(keys))]
    if (misc & (filePrefix == "")):
        colors = colorsBench[0:(len(keys)-1)] + colorMisc

    newKeys = augmentKeys(keys, values)
    # print(newKeys)
    print("Number of benchmarks: " +  str(np.sum(values)))
    if (filePrefix != ""):
        plt.rcParams["figure.figsize"] = (7.5,4.5)
    else:
        plt.rcParams["figure.figsize"] = (10,8)
    plotPiChartPrintPattern(newKeys, values, filePrefix+"benchmarkDistribution.pdf",patterns,colors)

    if (filePrefix != ""):
        return

    keys, values = filterToDetailSort(threshold,specifierMap,misc)
    newKeys = augmentKeys(keys, values)

    colors = colorsBench[0:(len(keys))]
    if (misc):
        colors = colorsBench[0:(len(keys)-1)] + colorMisc

    keysAuthor, keysX, resultData = authorInformation(True, authorThreshold, threshold, allSpecifier, specifierMap, True, False, anonymized,"No benchmark")
    plotAuthorInformation(keysAuthor, keysX, resultData, filePrefix+"benchmarkDistributionAuthor.pdf", patterns, (colors + colorNoEval), 10, 5)
    data = computeXPerConference(allSpecifier)
    dataFramesPlots = processXPerConference(data,keys,normalize,misc)
    for conf in conferences:
        plotDatePerYear(dataFramesPlots[conf],patterns,colors,conf,filePrefix+"benchmarkDistribution"+conf+".pdf",size1,size2,False,True)
        plotDatePerYear(dataFramesPlots[conf],patterns,colors,conf,filePrefix+"benchmarkDistribution"+conf+".pdf",size1,size2,False,False)
        if(usePdfCrop):
            subprocess.call(["pdfcrop", filePrefix+"leg_benchmarkDistribution"+conf+".pdf"])

def types(threshold=12,normalize=True,misc=True):
    size1 = 3
    size2 = 4

    if (filePrefix != ""):
        typeMapL, number = allTypes(df)
        thresholdL = 0
    else:
        typeMapL = typeMap
        thresholdL = threshold

    keys, values = filterToDetailSort(thresholdL,typeMapL,(misc & (filePrefix == "")))



    if (filePrefix != ""):
            keysG, _ = filterToDetailSort(threshold,typeMapGlobal,True)
            values = unifyKeys(keysG,keys,values)
            keys = keysG

    #          CaseStudy    Benchmark     TS Gen        Microbenc  DAG Gen      spec          N Gen   Misc
    patterns = ['**',          'OO',         '--',         'OOO',      '||',     '++',         '///', '']

    colors = colorsGlobal[0:(len(keys))]
    if (misc):
        colors = colorsGlobal[0:(len(keys)-1)] + colorMisc


    newKeys = augmentKeys(keys, values)
    plt.rcParams["figure.figsize"] = (5,3)
    plotPiChartPrintPattern(newKeys, values, filePrefix+"typeDistribution.pdf",patterns,colors)


    if (filePrefix != ""):
        return

    keys, values = filterToDetailSort(threshold,typeMap,misc)
    newKeys = augmentKeys(keys, values)

    colors = colorsGlobal[0:(len(keys))]
    if (misc):
        colors = colorsGlobal[0:(len(keys)-1)] + colorMisc

    keysAuthor, keysX, resultData = authorInformation(True, authorThreshold, threshold, allTypes, typeMap, True, False, anonymized,"No evaluation")
    plotAuthorInformation(keysAuthor, keysX, resultData, filePrefix+"typeDistributionAuthor.pdf", patterns, (colors[0:(len(keysX)-2)] + colorMiscAD + colorNoEval), 10, 3)
    data = computeXPerConference(allTypes)
    dataFramesPlots = processXPerConference(data,keys,normalize,misc)
    for conf in conferences:
        plotDatePerYear(dataFramesPlots[conf],patterns,colors,conf,filePrefix+"typeDistribution"+conf+".pdf",size1,size2,False,True)
        plotDatePerYear(dataFramesPlots[conf],patterns,colors,conf,filePrefix+"typeDistribution"+conf+".pdf",size1,size2,False,False)
        if(usePdfCrop):
            subprocess.call(["pdfcrop", filePrefix+"leg_typeDistribution"+conf+".pdf"])

def fields(threshold=12,normalize=True,misc=True):
    size1 = 3
    size2 = 4

    keys, values = filterToDetailSort(threshold,fieldMap,misc)

    patterns = ['','','','','','','','','']#['--',          'oo',        '//',          '\\\\',         '**',         'OO',      '++',          '..',         '']

    colors = colorsGlobal[0:(len(keys))]
    if (misc):
        colors = colorsGlobal[0:(len(keys)-1)] + colorMisc

    print("Number of fields: " +  str(np.sum(values)))
    newKeys = augmentKeys(keys, values)
    plt.rcParams["figure.figsize"] = (5,3)
    plotPiChartPrintPattern(newKeys, values, filePrefix+"fieldDistribution.pdf",patterns,colors)
    keysAuthor, keysX, resultData = authorInformation(True, authorThreshold, threshold, allFields, fieldMap, True, normalize, anonymized,"No case study")
    plotAuthorInformation(keysAuthor, keysX, resultData, filePrefix+"fieldDistributionAuthor.pdf", patterns, (colors[0:(len(keysX)-1)] + colorNoEval), 10, 3.25)
    data = computeXPerConference(allFields)
    dataFramesPlots = processXPerConference(data,keys,normalize,misc)
    # print(dataFramesPlots)
    for conf in conferences:
        plotDatePerYear(dataFramesPlots[conf],patterns,colors,conf,filePrefix+"fieldDistribution"+conf+".pdf",size1,size2,False,True)
        plotDatePerYear(dataFramesPlots[conf],patterns,colors,conf,filePrefix+"fieldDistribution"+conf+".pdf",size1,size2,False,False)
        if(usePdfCrop):
            subprocess.call(["pdfcrop", filePrefix+"leg_fieldDistribution"+conf+".pdf"])

def categories(threshold=0,normalize=True,misc=False):
    size1 = 3
    size2 = 4
    patterns = ('-', '+', '/', '\\', '*', 'o', 'O', '.')
    colors = colorsGlobal
    keys, values = filterToDetailSort(threshold,categoryMap,misc)
    newKeys = augmentKeys(keys, values)
    plt.rcParams["figure.figsize"] = (5,3)
    plotPiChartPrintPattern(newKeys, values, filePrefix+"categoryDistribution.pdf",patterns,colors)
    keysAuthor, keysX, resultData = authorInformation(True, authorThreshold, threshold, allCategories, categoryMap, False, False, anonymized,"no ???")
    plotAuthorInformation(keysAuthor, keysX, resultData, filePrefix+"categoryDistributionAuthor.pdf", patterns, (colors[0:(len(keysX)-1)] + colorNoEval), 10, 3)
    data = computeXPerConference(allCategories)
    dataFramesPlots = processXPerConference(data,keys,normalize,misc)
    # print(dataFramesPlots)
    for conf in conferences:
        plotDatePerYear(dataFramesPlots[conf],patterns,colors,conf,"categoryDistribution"+conf+".pdf",size1,size2,False,True)
        plotDatePerYear(dataFramesPlots[conf],patterns,colors,conf,"categoryDistribution"+conf+".pdf",size1,size2,False,False)
        if(usePdfCrop):
            subprocess.call(["pdfcrop", "leg_categoryDistribution"+conf+".pdf"])

def benchDiscardMalardalen():
    years = [2017,2018,2019,2020,2021,2022,2023,2024]
    plt.rcParams["figure.figsize"] = (10,2)
    amountMAL = computeQueryPerYear("MALARDALEN",allSpecifier,years)
    amountTB = computeQueryPerYear("TACLEBENCH",allSpecifier,years)
    plt.plot(years, amountMAL, label = "MALARDALEN (2005)",marker='x',color=colorsGlobal[0])
    plt.plot(years, amountTB, label = "TACLEBENCH (2016)",marker='o',color=colorsGlobal[-1])
    plt.ylim(bottom=0, top=8)
    plt.legend()
    plt.savefig(filePrefix+"discardBench.pdf", format="pdf", bbox_inches="tight")

def benchDiscardSDVB():
    years = [2017,2018,2019,2020,2021,2022,2023,2024]
    plt.rcParams["figure.figsize"] = (10,2)
    amountSD = computeQueryPerYear("SDVBS",allSpecifier,years)
    amountCS = computeQueryPerYear("CORTEXSUITE",allSpecifier,years)
    plt.plot(years, amountSD, label = "SDVBS (2009)",marker='x',color=colorsGlobal[0])
    plt.plot(years, amountCS, label = "CORTEXSUITE (2014)",marker='o',color=colorsGlobal[-1])
    plt.ylim(bottom=0, top=8)
    plt.legend()
    plt.savefig(filePrefix+"discardSDVD.pdf", format="pdf", bbox_inches="tight")

def benchDiscardPARSEC():
    years = [2017,2018,2019,2020,2021,2022,2023,2024]
    plt.rcParams["figure.figsize"] = (10,2)
    amountS2 = computeQueryPerYear("SPLASH2",allSpecifier,years)
    amountS2X = computeQueryPerYear("SPLASH2X",allSpecifier,years)
    amountS3 = computeQueryPerYear("SPLASH3",allSpecifier,years)
    amountPS = computeQueryPerYear("PARSEC",allSpecifier,years)
    plt.plot(years, amountS2, label = "SPLASH2 (deprecated)",marker='x')
    plt.plot(years, amountS2X, label = "SPLASH2X (deprecated)",marker='x')
    plt.plot(years, amountS3, label = "SPLASH3 (deprecated)",marker='x')
    plt.plot(years, amountPS, label = "PARSEC (new)",marker='o')
    plt.ylim(bottom=0, top=8)
    plt.legend()
    plt.savefig(filePrefix+"discardPARSEC.pdf", format="pdf", bbox_inches="tight")
    
def benchDiscardUUNIFASTDISCARD():
    years = [2017,2018,2019,2020,2021,2022,2023,2024]
    plt.rcParams["figure.figsize"] = (10,2)
    amountUD = computeQueryPerYear("UUNIFAST-DISCARD",allSpecifier,years)
    amountRFS = computeQueryPerYear("RANDFIXSUM",allSpecifier,years);
    amountDR = computeQueryPerYear("DIRICHLET-RESCALE",allSpecifier,years)
    plt.plot(years, amountUD, label = "UUNIFAST-DISCARD (2009)",marker='x',color=colorsGlobal[0])
    plt.plot(years, amountRFS, label = "RANDFIXSUM (2010)",marker='x',color=colorsGlobal[1])
    plt.plot(years, amountDR, label = "DIRICHLET-RESCALE (2020)",marker='o',color=colorsGlobal[-1])
    plt.ylim(bottom=0, top=8)
    plt.legend()
    plt.savefig(filePrefix+"discardUUniFastDiscard.pdf", format="pdf", bbox_inches="tight")


def nrOfEvals():
    keys = [0,1,2,3,4]
    data = []

    newkeys = []
    evals = 0

    for key in keys:
        amount= len(df.query('NROFEVAL == ' + str(key)))
        data.append(amount)
        newkeys.append(str(key) + " components ("+ str(amount) + ")")
        evals += amount*key

    evals += len(df.query('NROFEVAL == 5'))*5 + len(df.query('NROFEVAL == 6'))*6 + len(df.query('NROFEVAL == 7'))*7

    amount= len(df.query('NROFEVAL == 5')) + len(df.query('NROFEVAL == 6')) + len(df.query('NROFEVAL == 7'))
    data.append(amount)

    newkeys.append("between 4 and 7 components (" + str(amount) + ")")
    newkeys[0] = "no evaluation (" + str(len(df.query('NROFEVAL == 0'))) + ")"
    newkeys[1] = "one component (" + str(len(df.query('NROFEVAL == 1'))) + ")"

    patterns = ('','','','','','','')#('', '.', '..', '...', '....', '.....')
    colors = colorMisc + colorsGlobal

    plt.rcParams["figure.figsize"] = (5,3)
    plotPiChartPrintPattern(newkeys, data, "nrOfEvals.pdf",patterns,colors)

    print("Number of evals: " + str(evals))

    # return keys, data

def plotAll():
    benchDiscardMalardalen()
    plt.cla()
    plt.clf()
    benchDiscardSDVB()
    plt.cla()
    plt.clf()
    benchDiscardUUNIFASTDISCARD()
    plt.cla()
    plt.clf()
    reproducible(True,False)
    categories()
    benchmarks()
    types()
    fields()
    nrOfEvals()
    
    print("Number of unique specifiers: " + str(len(allSpecifier(df)[0].keys())))
    print("Number of unique specifiers only used once: " + str(len([k for k, v in dict(allSpecifier(df)[0]).items() if v == 1])))

    evalDivAuthors(authorThreshold,True)
    types()
    benchmarks()
    resetDF()

    evalDivAuthors(authorThreshold,False)
    types()
    benchmarks()
    resetDF()

def colorExtender (colors, nrKeys):
    result = []
    for c in colors:
        cExt = [c] * nrKeys
        result += cExt
    return result

def countNonEmptyEntries():
    columnIdentifiers = ['YEAR','AUTHOR1','AUTHOR2','AUTHOR3','AUTHOR4','AUTHOR5','AUTHOR6','AUTHOR7','AUTHOR8','AUTHOR9','AUTHOR10','AUTHOR11','AUTHOR12','AUTHOR13','AUTHOR14','AUTHOR15','TITLE','SPECIFIER1','QUALIFIER1','FIELD1','TYPE1','PROPRIETARY1','SPECIFIER2','QUALIFIER2','FIELD2','TYPE2','PROPRIETARY2','SPECIFIER3','QUALIFIER3','FIELD3','TYPE3','PROPRIETARY3','SPECIFIER4','QUALIFIER4','FIELD4','TYPE4','PROPRIETARY4','SPECIFIER5','QUALIFIER5','FIELD5','TYPE5','PROPRIETARY5','SPECIFIER6','QUALIFIER6','FIELD6','TYPE6','PROPRIETARY6','SPECIFIER7','QUALIFIER7','FIELD7','TYPE7','PROPRIETARY7','ARTIFACTEVALUATION','ARTIFACTWASAVAILABLE','ARTIFACTISAVAILABLE']


    totalSum = 0

    for column in columnIdentifiers:
        sum = len(df.query(column + ' != ""'))
        totalSum += sum
        # print(column + " " + str(sum) + " " + str(totalSum))

    return totalSum

def evalDivAuthors(threshold,setToAAbove):

    global df
    global typeMap
    global specifierMap
    global qualifierMap
    global fieldMap
    global filePrefix

    if (threshold > 1):
        authorsAboveT, otherAuthors = defineAuthorDF(threshold)
        filePrefix= "top"+str(threshold)+"-"
    else:
        authorsAboveT = dfComplete
        otherAuthors = pd.DataFrame()
        filePrefix= ""


    if (setToAAbove):
        df = authorsAboveT
    else:
        df = otherAuthors
        filePrefix += "below-"

    typeMap, number = allTypes(df)
    categoryMap, number = allCategories(df)
    specifierMap, number = allSpecifier(df)
    qualifierMap, number = allQualifier(df)
    fieldMap, number = allFields(df)

    print("Paper division: " + str(len(authorsAboveT)) + " paper above, " + str(len(otherAuthors)) + " paper below threshold of " + str(threshold) + " papers.")

    return authorsAboveT, otherAuthors

def resetDF():
    evalDivAuthors(0,True)
    filePrefix=""


# global Data structures
authorThreshold = 12
anonymized = True



filePrefix=""

typeMapGlobal = typeMap
categoryMapGlobal  = categoryMap
specifierMapGlobal = specifierMap
qualifierMapGlobal = qualifierMap
fieldMapGlobal = fieldMap

plotAll()
