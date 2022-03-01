from django.urls import path

from DB import views

urlpatterns = [
    path('index/',views.index,name='index'),
    path('indexCol/',views.indexCol,name='indexCol'),
    path('Search/',views.Search,name='Search'),
    path('SearchCol/',views.SearchCol,name='SearchCol'),
    path('Details/',views.Details,name='Details'),
    path('DetailsCol/',views.DetailsCol,name='DetailsCol'),
    path('userCommentCol/',views.userCommentCol,name='userCommentCol'),
    path('About/',views.About,name='About'),
    path('Download/',views.Download,name='Download'),
    path('Submit/',views.Submit,name='Submit'),
    path('SubmitCol/',views.SubmitCol,name='SubmitCol'),
    path('Contact/',views.Contact,name='Contact'),
    path('acqSeq/',views.acqSeq,name='acqSeq'),
]