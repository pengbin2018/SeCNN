
import preprocessing.get_improved_sbt as get_improved_sbt
import preprocessing.process_code as process_code
import preprocessing.process_nl as process_nl
import preprocessing.get_vocabulary as get_vocabulary


def start():
    get_improved_sbt.start()
    process_code.start()
    process_nl.start()
    print('generate vocabulary')
    get_vocabulary.start()