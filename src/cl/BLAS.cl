kernel void add(global float* a, global float* b, global float* c, int n) {
	int id = get_global_id(0);
	if(id < n) {
		c[id] = a[id] + b[id];
	}
}

kernel void sub(global float* a, global float* b, global float* c, int n) {
	int id = get_global_id(0);
	if(id < n) {
		c[id] = a[id] - b[id];
	}
}

kernel void mul(global float* a, global float* b, global float* c, int n) {
	int id = get_global_id(0);
	if(id < n) {
		c[id] = a[id] * b[id];
	}
}

kernel void div(global float* a, global float* b, global float* c, int n) {
	int id = get_global_id(0);
	if(id < n) {
		c[id] = a[id] / b[id];
	}
}

