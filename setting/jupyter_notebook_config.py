c = get_config()
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
c.NotebookApp.notebook_dir = '/home/jupyter'
c.NotebookApp.allow_origin = '*'
c.NotebookApp.token = ''
# 보안 위협에 노출 될 수 있으므로 반드시 password를 설정합니다. (sha)
c.NotebookApp.password = 'argon2:$argon2id$v=19$m=10240,t=10,p=8$xu0oar/J+4W9snVDNzFY7Q$Pm1nBtLxk/XzhPxOwEAePc5UDd8uZRlh3ULRn8MaoFw'
