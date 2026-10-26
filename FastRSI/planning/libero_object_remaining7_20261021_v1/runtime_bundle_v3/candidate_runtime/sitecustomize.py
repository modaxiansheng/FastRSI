import os
if os.environ.get('LIBEROPRO_BIND_ENV')=='1':
    try:
        from development_binding import install
        install()
    except BaseException:
        import traceback,sys
        traceback.print_exc();os._exit(93)
