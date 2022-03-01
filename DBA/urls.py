from django.urls import path

from DBA import views, self_info, level_manage, submit_msg, userCom_msg

urlpatterns = [
    path('index/',views.index,name='index'),
    path('indexCol/',views.indexCol,name='indexCol'),
    path('login/',views.login,name='login'),
    path('acquire_code/',views.acquire_code,name='acquire_code'),
    path('logout/',views.logout,name='logout'),
    path('self_center_update/',self_info.self_center_update,name='self_center_update'),
    path('level_manage_data/',level_manage.level_manage_data,name='level_manage_data'),
    path('level_manage_data_save/',level_manage.level_manage_data_save,name='level_manage_data_save'),
    path('level_manage_data_delete/',level_manage.level_manage_data_delete,name='level_manage_data_delete'),
    path('level_manage_data_add/',level_manage.level_manage_data_add,name='level_manage_data_add'),
    path('level_manage_data_del/',level_manage.level_manage_data_del,name='level_manage_data_del'),
    path('submit_msg_data/',submit_msg.submit_msg_data,name='submit_msg_data'),
    path('submit_msg_data_update/',submit_msg.submit_msg_data_update,name='submit_msg_data_update'),
    path('userCom_msg_data/',userCom_msg.userCom_msg_data,name='userCom_msg_data'),
    path('userCom_msg_data_update/',userCom_msg.userCom_msg_data_update,name='userCom_msg_data_update'),

]