#pragma once

class Marker
{
public:
	Marker(int id, float size, float margin);
	~Marker();

	inline const int ID() const { return id; }
	inline const float Size() const { return size; }
	inline const float Margin() const { return margin; }

private:
	const int id;
	const float size, margin;
};