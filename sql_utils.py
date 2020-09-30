# pylint: disable=invalid-name, redefined-outer-name, missing-docstring, non-parent-init-called, trailing-whitespace, line-too-long

import psycopg2

def postgree(conect_params,sql_query,query_params):
    try:
        connection=psycopg2.connect(**conect_params)
        cursor=connection.cursor()        
        cursor.execute(sql_query, query_params)
        connection.comit()
        cursor.close()
        connection.close()
        print('comited')
    except (Exception, psycopg2.Error) as error:
        if (connection):
            cursor.close()
            connection.close()
            

