extends Camera3D

var prevx : float
var prevy : float
var isMsAct : bool = false

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	prevx = get_viewport().get_mouse_position().x
	prevy = get_viewport().get_mouse_position().y


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:	
	
	if isMsAct == true:
		view()
	prevx = get_viewport().get_mouse_position().x 
	prevy = get_viewport().get_mouse_position().y 
	

func _input(event: InputEvent) -> void:
	if event.is_action("forward"):
		self.translate(Vector3(0,0,-1))
	if event.is_action("backward"):
		self.translate(Vector3(0,0,1))
	if event.is_action("rightmove"):
		self.translate(Vector3(1,0,0))
	if event.is_action("leftmove"):
		self.translate(Vector3(-1,0,0))
	if event.is_action_pressed("test"):
		isMsAct = !isMsAct
		print(isMsAct)
	
	
func view() -> void:
	self.global_rotate(Vector3(0,1,0),(get_viewport().get_mouse_position().x - prevx)/-50)
	self.rotate_object_local(Vector3(1,0,0),(get_viewport().get_mouse_position().y - prevy)/-50)
	
