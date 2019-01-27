#coding:utf-8
from bson.son import SON

def get_db():
	print "Entering get_db...        ",
	from pymongo import MongoClient
	client = MongoClient('localhost',27017)
	db = client.camboriu
	print("OK!")
	return db

def import_to_db(camboriu_entries_db):
	print "Entering import_to_db...        ",
	import xml.etree.ElementTree as ET
	parser = ET.XMLParser(encoding="utf-8")
	parser.parser.UseForeignDTD(True)
	etree = ET.ElementTree()
	camboriu_xml_root = etree.parse('camboriu.osm',parser=parser)
	
	
	for node in camboriu_xml_root.findall('node'):
		json_entry = {}
		has_name = False
		for tag in node.findall('tag'):
			if(tag.get('k')=="name"):
				json_entry['name'] = tag.get('v').encode('utf-8')
				json_entry['author'] = node.get('user').encode('utf-8')
				json_entry['lat'] = node.get('lat').encode('utf-8')
				json_entry['lon'] = node.get('lon').encode('utf-8')
				has_name = True

			else:
				json_entry[tag.get('k').encode('utf-8')] = tag.get('v').encode('utf-8')
				
		if has_name:
			entry_id = camboriu_entries_db.insert_one(json_entry).inserted_id
			print("Added to DB: " + str(entry_id))	
	

	for way in camboriu_xml_root.findall('way'):
		json_entry = {}
		has_name = False
		for tag in way.findall('tag'):
			if(tag.get('k')=="name"):
				json_entry['name'] = tag.get('v').encode('utf-8')
				json_entry['author'] = way.get('user').encode('utf-8')
				if(way.get('lat') is not None):
					json_entry['lat'] = way.get('lat').encode('utf-8')
				if(way.get('lon') is not None):
					json_entry['lon'] = way.get('lon').encode('utf-8')
				has_name = True

			else:
				json_entry[tag.get('k').encode('utf-8')] = tag.get('v').encode('utf-8')
		if has_name:
			entry_id = camboriu_entries_db.insert_one(json_entry).inserted_id
			print("Added to DB: " + str(entry_id))	


def list_cities(camboriu_entries_db):
	pipeline = [
     {"$unwind": "$addr:city"},
     {"$group": {"_id": "$addr:city", "count": {"$sum": 1}}},
     {"$sort": SON([("count", -1), ("_id", -1)])}
	]
	entries_with_cities = list(camboriu_entries_db.aggregate(pipeline))
	for entry in entries_with_cities:
		print unicode(entry['_id']+'  ').encode('utf-8'),
		print unicode(entry['count'])

def top10_user_contribution(camboriu_entries_db):
	pipeline = [
     {"$unwind": "$author"},
     {"$group": {"_id": "$author", "count": {"$sum": 1}}},
     {"$sort": SON([("count", -1), ("_id", -1)])},
     {"$limit":10}
	]
	author_contribution_list = list(camboriu_entries_db.aggregate(pipeline))
	for contrib in author_contribution_list:
		print unicode(contrib['_id']+'  ').encode('utf-8'),
		print unicode(contrib['count'])

def count_authors(camboriu_entries_db):
	print(len(camboriu_entries_db.distinct('author')))

def top10_amenities(camboriu_entries_db):
	pipeline = [
     {"$unwind": "$amenity"},
     {"$group": {"_id": "$amenity", "count": {"$sum": 1}}},
     {"$sort": SON([("count", -1), ("_id", -1)])},
     {"$limit":10}
	]
	author_contribution_list = list(camboriu_entries_db.aggregate(pipeline))
	for contrib in author_contribution_list:
		print unicode(contrib['_id']+'  ').encode('utf-8'),
		print unicode(contrib['count'])

def top10_cuisines(camboriu_entries_db):
	pipeline = [
     {"$unwind": "$cuisine"},
     {"$group": {"_id": "$cuisine", "count": {"$sum": 1}}},
     {"$sort": SON([("count", -1), ("_id", -1)])},
     {"$limit":10}
	]
	author_contribution_list = list(camboriu_entries_db.aggregate(pipeline))
	for contrib in author_contribution_list:
		print unicode(contrib['_id']+'  ').encode('utf-8'),
		print unicode(contrib['count'])

def top3_religion(camboriu_entries_db):
	pipeline = [
     {"$unwind": "$religion"},
     {"$group": {"_id": "$religion", "count": {"$sum": 1}}},
     {"$sort": SON([("count", -1), ("_id", -1)])},
     {"$limit":3}
]
	author_contribution_list = list(camboriu_entries_db.aggregate(pipeline))
	for contrib in author_contribution_list:
		print unicode(contrib['_id']+'  ').encode('utf-8'),
		print unicode(contrib['count'])


def del_db():
	from pymongo import MongoClient
	client = MongoClient('localhost',27017)
	client.drop_database('camboriu')


	print("OK!")
if __name__ == '__main__':
	
	camboriu_db = get_db()
	camboriu_entries_db = camboriu_db.camboriu_entries_db
	#import_to_db(camboriu_entries_db)
	#list_cities(camboriu_entries_db)
	#top10_user_contribution(camboriu_entries_db)
	#count_authors(camboriu_entries_db)
	#top10_amenities(camboriu_entries_db)
	#top10_cuisines(camboriu_entries_db)
	top3_religion(camboriu_entries_db)
