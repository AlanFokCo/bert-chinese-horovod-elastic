import pymysql as mysql
import logging
import os

class MysqlHandler():

    def __init__(self, host, port, user, password):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.checkDB(host=host, port=port, user=user, password=password)

    def checkDB(self, host, port, user, password):
        try:
            connection = mysql.connect(host=host, port=port, user=user, password=password, db='kubeai')
            self.checkTable(connection)
            connection.close()
        except Exception as e:
            if e.args[0] == 1049:
                connection = mysql.connect(host=host, port=port, user=user, password=password)
                create_db = "CREATE DATABASE kubeai"
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(create_db)
                    connection.commit()
                    self.checkTable(connection)
                    connection.close()
                except Exception as e:
                    logging.error(str(e.args))
            else:
                logging.error(str(e.args))

    def checkTable(self, connection):
        check_table = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='kubeai' AND TABLE_NAME='evaluate';"
        try:
            with connection.cursor() as cursor:
                cursor.execute(check_table)
                result = cursor.fetchall()
                if len(result) == 0:
                    create_table = '''
                    CREATE TABLE kubeai.evaluate(
                        id bigint(20) not null auto_increment primary key,
                        job_id varchar(50) not null,
                        name varchar(256),
                        namespace varchar(256),
                        uid varchar(256),
                        model_id varchar(256),
                        status varchar(32),
                        image varchar(256),
                        dataset_path varchar(256),
                        code varchar(256),
                        command varchar(256),
                        metrics text,
                        is_deleted tinyint(4),
                        report_path varchar(256),
                        gmt_created datetime,
                        gmt_modified datetime
                    );
                    '''
                    try:
                        with connection.cursor() as cursor:
                            cursor.execute(create_table)
                        connection.commit()
                    except Exception as e:
                        logging.error(str(e.args))
        except Exception as e:
            logging.error(str(e.args))

    def writeMetrics(self, metrics):
        connection = mysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, db='kubeai')
        print(connection)
        self.job_id = os.getenv('JOB_ID')
        self.dataset_dir = os.getenv('DATASET_PATH')
        self.report_dir = os.getenv('METRICS_PATH')
        select_message = '''SELECT COUNT(*) as numbers FROM kubeai.evaluate WHERE job_id='%s';''' % str(self.job_id)
        try:
            with connection.cursor() as cursor:
                cursor.execute(select_message)
                result = cursor.fetchall()
                if result[0][0] == 0:
                    self.insertMetrics(connection, metrics)
                else:
                    self.updateMetrics(connection, metrics)
        except Exception as e:
            logging.error(str(e.args))

    def insertMetrics(self, connection, metrics):
        update_message = '''INSERT INTO kubeai.evaluate(
                                job_id,
                                dataset_path,
                                metrics,
                                report_path
                            ) VALUES (
                                "%s",
                                "%s",
                                "%s",
                                "%s"
                            );''' % (str(self.job_id) , self.dataset_dir, str(metrics), self.report_dir)
        try:
            with connection.cursor() as cursor:
                cursor.execute(update_message)
            connection.commit()
        except Exception as e:
            logging.error(str(e.args))

    def updateMetrics(self, connection, metrics):
        update_message = '''UPDATE kubeai.evaluate SET metrics = "%s" WHERE job_id="%s";''' % (str(metrics), str(self.job_id))
        try:
            with connection.cursor() as cursor:
                cursor.execute(update_message)
            connection.commit()
        except Exception as e:
            logging.error(str(e.args))