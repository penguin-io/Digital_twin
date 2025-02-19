extends Node

var empLst : Array[MeshInstance3D]
var lbLst : Array[Label3D]
var tempp : Dictionary

var ind : int = 0

@export var empPrefab : MeshInstance3D
@export var lb : Label
@export var cam: Camera3D

@export var timer : Timer


var dbDir : String # Database location path varaible

var isTimerActive = false # boolean to check timer status

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	dbDir = OS.get_executable_path().get_base_dir() + "/camera1.json"
	
	
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	lb.text = "FPS : " + str(1/delta)
	if isTimerActive == false:
		timer.start()
		isTimerActive = true

func read(index : int) -> Array:
	var testDir = "/home/thiru/GODOT/Projects/Digital_twin/Main/camera1.json"
	var file = FileAccess.open(testDir,FileAccess.READ)
	tempp = JSON.parse_string(file.get_as_text())
	if index <= len(tempp["frames"]):
		Global.content = tempp["frames"][index]["persons"]
		print(index)
	else:
		Global.content = []
	return Global.content

func instantiate(db):
	var offset = len(db) - len(empLst)
	if offset == 0 and len(db) != 0:
		pass
	elif offset > 0:
		for i in range(offset):
			var tempEmp = MeshInstance3D.new()
			tempEmp.mesh = empPrefab.mesh
			tempEmp.material_override = empPrefab.material_override
			tempEmp.position.y = empPrefab.position.y
			tempEmp.scale = empPrefab.scale
			add_child(tempEmp)
			empLst.append(tempEmp)	
			
	elif offset < 0:
		for i in range(-offset):
			if empLst.size() != 0:
				var temp : MeshInstance3D = empLst.pop_at(-1)
				temp.queue_free()

func labelInstatiate():
	for i in range(len(empLst)):
		var templb3d = Label3D.new()
		templb3d.text = "Employee : \n" + str(Global.content[i]["person_id"])
		templb3d.position.y += 1.5
		templb3d.font_size = 150
		templb3d.rotation = cam.rotation
		empLst[i].add_child(templb3d)
		lbLst.append(templb3d)

func _on_timer_timeout() -> void:
	instantiate(read(Global.ind))
	updatePos(read(Global.ind))
	
	for lb in lbLst:
		lb.queue_free()
	lbLst.clear()
	
	labelInstatiate()
	isTimerActive = false
	
	Global.ind += 1
	
func updatePos(db):
	if len(empLst) != 0:
		for i in range(len(empLst)):
			empLst[i].position.x = (db[i]["center_point"]["relative"]["x"]) * 0.1
			empLst[i].position.z = (db[i]["center_point"]["relative"]["y"]) * 0.1
	
