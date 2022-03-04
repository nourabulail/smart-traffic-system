from django.shortcuts import render
from django.db import models
from django.db.models import Count, Sum
from django.db.models.functions import TruncYear, TruncMonth, TruncDay, TruncHour
from django.http import HttpResponse
import django_tables2 as tables
import MySQLdb
from django_tables2.config import RequestConfig
import itertools
from django.db import connection
from djqscsv import render_to_csv_response
from datetime import datetime
from pytz import timezone

##### Modify with your database here #####
# db = MySQLdb.connect("localhost", "root", "", "detect_car_db", charset='utf8')
# cursor = db.cursor()

category_list = ['Year', 'Month', 'Day', 'Hour']

class vehicle_info(models.Model):
    camera_id = models.CharField(max_length=24)
    pass_count = models.IntegerField(default=0)
    accident_count = models.IntegerField(default=0)
    time = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "vehicle_info"


class vehicleInfoTable(tables.Table):
    counter = tables.Column(verbose_name="No", empty_values=(), orderable=False)
    time = tables.DateTimeColumn(verbose_name="Time", format ='Y-m-d H:i:s')
    pass_count = tables.Column(verbose_name="Car count")
    camera_id = tables.Column(verbose_name="Camera Location")
    accident_count = tables.Column(verbose_name="Accident Count")

    def render_counter(self):
        self.row_counter = getattr(self, 'row_counter', itertools.count(1))
        return next(self.row_counter)

    class Meta:
        model = vehicle_info
        attrs = {
            "class": "info-table",
        }
        fields = ("counter","camera_id", "time", "pass_count", "accident_count" )

class vehicleInfoByCameraTable(tables.Table):
    counter = tables.Column(verbose_name="No", empty_values=(), orderable=False)
    time = tables.DateTimeColumn(verbose_name="Time", format ='Y-m-d H:i:s')
    pass_count = tables.Column(verbose_name="Car count")
    accident_count = tables.Column(verbose_name="Accident count")

    def render_counter(self):
        self.row_counter = getattr(self, 'row_counter', itertools.count(1))
        return next(self.row_counter)
    class Meta:
        model = vehicle_info
        attrs = {
            "class": "info-table",
        }
        fields = ("counter", "time", "pass_count", "accident_count")

class filterTable(tables.Table):
    counter = tables.Column(verbose_name="No", empty_values=(), orderable=False)
    filter_time = tables.DateTimeColumn(verbose_name="Time", format ='Y-m-d H:i:s')
    pass_count = tables.Column(verbose_name="Car Count")
    camera_id = tables.Column(verbose_name="Camera Location")
    accident_count = tables.Column(verbose_name="Accident Count")

    def render_counter(self):
        self.row_counter = getattr(self, 'row_counter', itertools.count(1))
        return next(self.row_counter)

    class Meta:
        model = vehicle_info
        attrs = {
            "class": "info-table",
        }
        fields = ("counter","camera_id", "filter_time", "pass_count", "accident_count")

class filterByCameraTable(tables.Table):
    counter = tables.Column(verbose_name="No", empty_values=(), orderable=False)
    filter_time = tables.DateTimeColumn(verbose_name="Time", format ='Y-m-d H:i:s')
    pass_count = tables.Column(verbose_name="Vehicles")
    accident_count = tables.Column(verbose_name="Accident Count")

    def render_counter(self):
        self.row_counter = getattr(self, 'row_counter', itertools.count(1))
        return next(self.row_counter)
    class Meta:
        model = vehicle_info
        attrs = {
            "class": "info-table",
        }
        fields = ("counter", "filter_time", "pass_count", "accident_count")

global_timezone = timezone('America/Fortaleza')
def save_vehicles(request):
    cameraid = request.GET['camera_id']
    passcount = request.GET['pass_count']
    accidentcount = request.GET['accident_count']
    current_time = datetime.now().replace(tzinfo=global_timezone) #tz aware
    v_info = vehicle_info(camera_id = cameraid, pass_count = passcount, accident_count=accidentcount, time=current_time)
    v_info.save()
    return HttpResponse("Success")


def to_render(html_render, cameras, table):
    html_render['table'] = table
    html_render['category_list'] = category_list
    html_render['cameras'] = cameras


def traffic(request):
    data = vehicle_info.objects.all()
    data = data.values('time', 'pass_count', 'camera_id', 'accident_count')
    cameras = (vehicle_info.objects.values('camera_id').annotate(ccount=Count('camera_id')).order_by('camera_id'))
    table = vehicleInfoTable(data) 
    RequestConfig(request, paginate={'per_page': 100}).configure(table)
    html_render = {}
    to_render(html_render, cameras, table)
    return render(request, "index.html", html_render)

# rendering "Filter"
def traffic_filter(request):
    html_render = {}
    cameras = (vehicle_info.objects.values('camera_id').annotate(ccount=Count('camera_id')).order_by('camera_id'))

    if request.GET['filter_time'] == '':
        data = vehicle_info.objects.all()
        if request.GET['filter_camera'] == 'all':
            data = data.values("time", "pass_count", "camera_id", "accident_count")
            table = vehicleInfoTable(data)
            RequestConfig(request, paginate={'per_page': 100}).configure(table)
            to_render(html_render, cameras, table)
        else:
            data = data.filter(models.Q(camera_id__icontains=request.GET['filter_camera']))
            data = data.values("time", "pass_count", "accident_count")
            table = vehicleInfoByCameraTable(data)
            RequestConfig(request, paginate={'per_page': 100}).configure(table)
            to_render(html_render, cameras, table)
        if request.GET['filter_camera'] != 'all':
            html_render['filter_camera'] = (request.GET['filter_camera'])
        html_render['filter_time'] = request.GET['filter_time']
        return render(request, "index.html", html_render)
    
    if request.GET['filter_time'] == 'Year':
        data = vehicle_info.objects.annotate(filter_time=TruncYear('time')).values('filter_time', 'camera_id').annotate(pass_count=Sum('pass_count')).annotate(accident_count=Sum('accident_count')).order_by('filter_time')
    if request.GET['filter_time'] == 'Month':
        data = vehicle_info.objects.annotate(filter_time=TruncMonth('time')).values('filter_time', 'camera_id').annotate(pass_count=Sum('pass_count')).annotate(accident_count=Sum('accident_count')).order_by('filter_time')
    if request.GET['filter_time'] == 'Day':
        data = vehicle_info.objects.annotate(filter_time=TruncDay('time')).values('filter_time', 'camera_id').annotate(pass_count=Sum('pass_count')).annotate(accident_count=Sum('accident_count')).order_by('filter_time')
    if request.GET['filter_time'] == 'Hour':
        data = vehicle_info.objects.annotate(filter_time=TruncHour('time')).values('filter_time', 'camera_id').annotate(pass_count=Sum('pass_count')).annotate(accident_count=Sum('accident_count')).order_by('filter_time')
    
    
    if request.GET['filter_camera'] == 'all':
        data = data.values("filter_time", "pass_count", "camera_id", "accident_count")
        print(data)
        table = filterTable(data)
        RequestConfig(request, paginate={'per_page': 100}).configure(table)
        to_render(html_render, cameras, table)
    else:
        data = data.filter(models.Q(camera_id__icontains=request.GET['filter_camera']))
        data = data.values("filter_time", "pass_count", "accident_count")
        table = filterByCameraTable(data)
        RequestConfig(request, paginate={'per_page': 100}).configure(table)
        to_render(html_render, cameras, table)
    
    if request.GET['filter_camera'] != 'all':
            html_render['filter_camera'] = (request.GET['filter_camera'])
    html_render['filter_time'] = request.GET['filter_time']

    return render(request, "index.html", html_render)
    